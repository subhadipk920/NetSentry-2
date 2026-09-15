"""
NetSentry Wire Traffic Simulator (Hacker vs Benign User).
Tests NetSentry reverse-proxy gateway and direct ML inference API on live Modal deployment.
"""

import os
import json
import requests

BASE_URL = os.getenv("NETSENTRY_SERVER_URL", "https://dipk6545--netsentry-serving-fastapi-app.modal.run")
GATEWAY_URL = f"{BASE_URL}/gateway/get"
PREDICT_URL = f"{BASE_URL}/v1/predict"

def test_gateway_benign_user():
    print("\n" + "=" * 60)
    print("1. GATEWAY SIMULATION: LEGITIMATE BENIGN USER REQUEST")
    print(f"Target: {GATEWAY_URL}")
    print("=" * 60)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json",
    }
    try:
        resp = requests.get(GATEWAY_URL, headers=headers, timeout=15)
        print(f"Status Code: {resp.status_code}")
        if resp.status_code == 200:
            print("✅ PASSED: Legitimate user granted access through gateway to upstream server!")
        else:
            print(f"Response: {resp.text[:200]}")
    except Exception as e:
        print(f"Connection Error: {e}")

def test_gateway_syn_scan_attacker():
    print("\n" + "=" * 60)
    print("2. GATEWAY SIMULATION: HACKER SYN PORT SCAN / RECON")
    print(f"Target: {GATEWAY_URL}")
    print("=" * 60)
    headers = {
        "User-Agent": "Nmap NSE / Raw Socket Scanner",
        "X-Flag": "syn",
        "X-Attack": "scan",
    }
    try:
        resp = requests.get(GATEWAY_URL, headers=headers, timeout=15)
        print(f"Status Code: {resp.status_code}")
        print(f"Response: {json.dumps(resp.json(), indent=2)}")
        if resp.status_code == 403:
            print("🛡️ PASSED: NetSentry intercepted and BLOCKED the SYN scanner at the gateway!")
        else:
            print("⚠️ ALERT: Malicious scan bypassed the shield!")
    except Exception as e:
        print(f"Connection Error: {e}")

def test_gateway_dos_flood_attacker():
    print("\n" + "=" * 60)
    print("3. GATEWAY SIMULATION: HIGH-RATE EXPLOIT PAYLOAD BURST (DOS)")
    print(f"Target: {GATEWAY_URL}")
    print("=" * 60)
    headers = {
        "User-Agent": "Slowloris / HighRate DOS Engine",
        "X-Attack": "flood",
    }
    try:
        resp = requests.get(GATEWAY_URL, headers=headers, timeout=15)
        print(f"Status Code: {resp.status_code}")
        print(f"Response: {json.dumps(resp.json(), indent=2)}")
        if resp.status_code == 403:
            print("🛡️ PASSED: NetSentry intercepted and BLOCKED exploit payload burst!")
        else:
            print("⚠️ ALERT: Exploit payload bypassed the shield!")
    except Exception as e:
        print(f"Connection Error: {e}")

def test_direct_ml_predict_api():
    print("\n" + "=" * 60)
    print("4. DIRECT REST API INFERENCE: POST /v1/predict")
    print(f"Target: {PREDICT_URL}")
    print("=" * 60)
    
    # Representative attack flow vector (high packet density, asymmetrical byte flow)
    payload = {
        "features": {
            "Destination_Port": 80,
            "Protocol": 6.0,
            "Flow_Duration": 120000,
            "Total_Fwd_Packets": 45,
            "Total_Backward_Packets": 2,
            "Total_Length_of_Fwd_Packets": 15000,
            "Total_Length_of_Bwd_Packets": 100,
            "Fwd_Packet_Length_Max": 1460,
            "Fwd_Packet_Length_Min": 64,
            "Fwd_Packet_Length_Mean": 333.3,
            "Fwd_Packet_Length_Std": 120.5,
            "Bwd_Packet_Length_Max": 100,
            "Bwd_Packet_Length_Min": 0,
            "Bwd_Packet_Length_Mean": 50.0,
            "Bwd_Packet_Length_Std": 10.0,
            "Flow_Bytes_s": 125833.3,
            "Flow_Packets_s": 391.6,
            "Flow_IAT_Mean": 2600.0,
            "Flow_IAT_Std": 1200.0,
            "Flow_IAT_Max": 8000.0,
            "Flow_IAT_Min": 50.0,
            "Fwd_IAT_Total": 118000.0,
            "Fwd_IAT_Mean": 2680.0,
            "Fwd_IAT_Std": 1100.0,
            "Fwd_IAT_Max": 8000.0,
            "Fwd_IAT_Min": 50.0,
            "Bwd_IAT_Total": 5000.0,
            "Bwd_IAT_Mean": 2500.0,
            "Bwd_IAT_Std": 100.0,
            "Bwd_IAT_Max": 3000.0,
            "Bwd_IAT_Min": 2000.0,
            "Fwd_PSH_Flags": 1,
            "Bwd_PSH_Flags": 0,
            "Fwd_URG_Flags": 0,
            "Bwd_URG_Flags": 0,
            "Fwd_Header_Length": 900,
            "Bwd_Header_Length": 64,
            "Fwd_Packets_s": 375.0,
            "Bwd_Packets_s": 16.6,
            "Min_Packet_Length": 0,
            "Max_Packet_Length": 1460,
            "Packet_Length_Mean": 321.2,
            "Packet_Length_Std": 210.4,
            "Packet_Length_Variance": 44268.0,
            "FIN_Flag_Count": 0,
            "SYN_Flag_Count": 1,
            "RST_Flag_Count": 0,
            "PSH_Flag_Count": 1,
            "ACK_Flag_Count": 1,
            "URG_Flag_Count": 0,
            "CWE_Flag_Count": 0,
            "ECE_Flag_Count": 0,
            "Down_Up_Ratio": 0.04,
            "Average_Packet_Size": 328.0,
            "Avg_Fwd_Segment_Size": 333.3,
            "Avg_Bwd_Segment_Size": 50.0,
            "Init_Win_bytes_forward": 29200,
            "Init_Win_bytes_backward": 28960,
            "act_data_pkt_fwd": 30,
            "min_seg_size_forward": 20,
            "Active_Mean": 0.0,
            "Active_Std": 0.0,
            "Active_Max": 0.0,
            "Active_Min": 0.0,
            "Idle_Mean": 0.0,
            "Idle_Std": 0.0,
            "Idle_Max": 0.0,
            "Idle_Min": 0.0
        }
    }

    try:
        resp = requests.post(PREDICT_URL, json=payload, timeout=15)
        print(f"Status Code: {resp.status_code}")
        print(f"Response: {json.dumps(resp.json(), indent=2)}")
        if resp.status_code == 200:
            result = resp.json()
            verdict = "🚨 ATTACK" if result.get("prediction") == 1 else "✅ BENIGN"
            print(f"Prediction Verdict: {verdict} (Probability: {result.get('probability'):.4f})")
            print(f"Model Version: {result.get('model_version')}")
        else:
            print("❌ Prediction request failed.")
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    print(f"🚀 NetSentry Live Cloud Traffic Simulator targeting: {BASE_URL}")
    test_gateway_benign_user()
    test_gateway_syn_scan_attacker()
    test_gateway_dos_flood_attacker()
    test_direct_ml_predict_api()
