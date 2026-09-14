"""
NetSentry Wire Traffic Simulator (Hacker vs Benign User).
Tests NetSentry reverse-proxy gateway against real malicious & benign network signatures.
"""

import time
import requests
import json

GATEWAY_URL = "http://127.0.0.1:8000/gateway/x"

def test_benign_user():
    print("\n" + "=" * 60)
    print("1. SIMULATING LEGITIMATE (BENIGN) USER TRAFFIC")
    print("=" * 60)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json",
    }
    try:
        resp = requests.get(GATEWAY_URL, headers=headers, timeout=5)
        print(f"Status Code: {resp.status_code}")
        print(f"Response: {resp.json()}")
        if resp.status_code == 200:
            print("✅ PASSED: Legitimate user granted access to friend's server!")
        else:
            print("❌ FAILED: Legitimate traffic was blocked unexpectedly.")
    except Exception as e:
        print(f"Connection Error: {e}")

def test_syn_scan_attacker():
    print("\n" + "=" * 60)
    print("2. SIMULATING HACKER: SYN PORT SCANNER / RECONNAISSANCE")
    print("=" * 60)
    headers = {
        "User-Agent": "Nmap NSE / Raw Socket Scanner",
        "X-Flag": "syn",
        "X-Attack": "scan",
    }
    try:
        resp = requests.get(GATEWAY_URL, headers=headers, timeout=5)
        print(f"Status Code: {resp.status_code}")
        print(f"Response: {json.dumps(resp.json(), indent=2)}")
        if resp.status_code == 403:
            print("🛡️ BLOCKED: NetSentry intercepted and blocked the SYN scanner!")
        else:
            print("⚠️ ALERT: Malicious scan bypassed the shield!")
    except Exception as e:
        print(f"Connection Error: {e}")

def test_dos_flood_attacker():
    print("\n" + "=" * 60)
    print("3. SIMULATING HACKER: HIGH-RATE EXPLOIT PAYLOAD BURST")
    print("=" * 60)
    headers = {
        "User-Agent": "Slowloris / HighRate DOS Engine",
        "X-Attack": "flood",
    }
    try:
        resp = requests.get(GATEWAY_URL, headers=headers, timeout=5)
        print(f"Status Code: {resp.status_code}")
        print(f"Response: {json.dumps(resp.json(), indent=2)}")
        if resp.status_code == 403:
            print("🛡️ BLOCKED: NetSentry intercepted and blocked payload burst!")
        else:
            print("⚠️ ALERT: Exploit payload bypassed the shield!")
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    print("🚀 NetSentry Gateway Traffic Simulator")
    test_benign_user()
    test_syn_scan_attacker()
    test_dos_flood_attacker()
