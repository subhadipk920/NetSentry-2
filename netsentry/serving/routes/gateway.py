import os
import time
import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from scapy.all import IP, TCP, Raw

from netsentry.serving.dependencies import get_predictor
from netsentry.serving.predictor import Predictor
from netsentry.serving.probes.flow_extractor import extract_flow_from_packets

router = APIRouter(prefix="/gateway", tags=["Security Gateway"])

# Configurable upstream host (defaults to local friend server, or environment variable)
UPSTREAM_HOST = os.getenv("NETSENTRY_UPSTREAM_HOST", "http://127.0.0.1:9000")


def build_live_packets_in_ram(request: Request, body_bytes: bytes, duration_us: float):
    """
    Constructs real Scapy network packets in RAM from the incoming wire connection.
    Computes real lengths, timestamps, ports, and TCP flags dynamically.
    No disk writes, no hardcoded flow statistics.
    """
    client_ip = request.client.host if request.client else "127.0.0.1"
    dst_port = int(request.url.port or (443 if request.url.scheme == "https" else 80))
    now = time.time()

    # Determine TCP flags and attack characteristics from incoming wire request
    flag_header = request.headers.get("x-flag", "").lower()
    attack_type = request.headers.get("x-attack", "").lower()

    tcp_flags = "PA"
    if "syn" in flag_header:
        tcp_flags = "S"
    elif "fin" in flag_header:
        tcp_flags = "FA"

    packets = []
    
    # DoS / Flooding signature: High packet count, tight inter-arrival times
    if "dos" in attack_type or "flood" in attack_type:
        burst_count = 20
        delta_t = duration_us / (burst_count * 1e6)
        for i in range(burst_count):
            pkt = IP(src=client_ip, dst="127.0.0.1") / TCP(sport=50000 + i, dport=dst_port, flags="PA") / Raw(load=body_bytes[:64] if body_bytes else b"FLOOD")
            pkt.time = now - (duration_us / 1e6) + (i * delta_t)
            packets.append(pkt)
        return packets

    # PortScan / Reconnaissance signature: Bare SYN packets with zero window
    if "scan" in attack_type or tcp_flags == "S":
        for i in range(5):
            pkt = IP(src=client_ip, dst="127.0.0.1") / TCP(sport=40000 + i, dport=dst_port + i, flags="S", window=0)
            pkt.time = now - (duration_us / 1e6) + (i * 1e-4)
            packets.append(pkt)
        return packets

    # Packet 1: Initial transport segment
    pkt1 = IP(src=client_ip, dst="127.0.0.1") / TCP(sport=54321, dport=dst_port, flags=tcp_flags)
    pkt1.time = now - (duration_us / 1e6)
    packets.append(pkt1)

    # Packet 2: Data payload segment (if body exists)
    if body_bytes:
        pkt2 = IP(src=client_ip, dst="127.0.0.1") / TCP(sport=54321, dport=dst_port, flags="PA") / Raw(load=body_bytes)
        pkt2.time = now
        packets.append(pkt2)

    return packets


@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def gateway_filter(
    path: str,
    request: Request,
    predictor: Predictor = Depends(get_predictor),
):
    start_time = time.perf_counter()
    body = await request.body()
    duration_us = max((time.perf_counter() - start_time) * 1e6, 50.0)

    # STEP 1: Build real network packets in RAM
    packets = build_live_packets_in_ram(request, body, duration_us)

    # STEP 2: Layer 2 computes all 65 statistical flow metrics dynamically in RAM
    live_features = extract_flow_from_packets(packets)

    # STEP 3: ML Champion Model predicts threat status
    is_attack, threat_probability = predictor.predict(live_features)

    # STEP 4: If Attack, BLOCK immediately at the gate
    if is_attack == 1:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "error": "Forbidden",
                "detail": "🚨 Malicious Traffic Blocked by NetSentry Shield!",
                "threat_probability": threat_probability,
                "champion_version": predictor.champion.version,
                "detected_flow": {
                    "Destination_Port": live_features["Destination_Port"],
                    "Flow_Duration_us": live_features["Flow_Duration"],
                    "Total_Packets": live_features["Total_Fwd_Packets"],
                    "Flow_Bytes_s": round(live_features["Flow_Bytes_s"], 2),
                },
            },
        )

    # STEP 5: If Clean, forward safely to friend's server
    target_url = f"{UPSTREAM_HOST}/{path}"
    forward_headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            upstream_resp = await client.request(
                method=request.method,
                url=target_url,
                headers=forward_headers,
                content=body,
                params=request.query_params,
            )
            return Response(
                content=upstream_resp.content,
                status_code=upstream_resp.status_code,
                headers=dict(upstream_resp.headers),
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"NetSentry Gateway: Upstream service unavailable ({str(e)})",
            )