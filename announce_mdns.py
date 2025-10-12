"""Announce an IPv4 address as specsserver.local using mDNS (zeroconf).

Usage:
    python announce_mdns.py --ip 192.168.1.42 --port 80

The script publishes an A record for specsserver.local and a simple HTTP service _http._tcp.local.
Ctrl-C to stop.
"""
import argparse
import socket
import time
from zeroconf import IPVersion, ServiceInfo, Zeroconf, NonUniqueNameException


def build_service_info(name: str, ip: str, port: int) -> ServiceInfo:
    # zeroconf expects bytes for text records and addresses as packed bytes
    addresses = [socket.inet_aton(ip)]
    desc = {b'path': b'/'}
    service_name = f"{name}._http._tcp.local."  # visible service name

    info = ServiceInfo(
        type_="_http._tcp.local.",
        name=service_name,
        addresses=addresses,
        port=port,
        properties=desc,
        server=f"{name}.local.",
    )
    return info


def main():
    parser = argparse.ArgumentParser(description="Announce specsserver.local via mDNS")
    parser.add_argument("--ip", required=False, help="IPv4 address to advertise (e.g. 192.168.1.42). If omitted the script will try to auto-detect the primary IPv4.")
    parser.add_argument("--port", type=int, default=80, help="Port for the HTTP service (default: 80)")
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument("--name", default="specsserver", help="Base name to advertise (default: specsserver)")
    args = parser.parse_args()

    # Basic validation
    def detect_local_ip() -> str:
        """Return the primary IPv4 address for outbound connections.

        This uses a UDP socket to a public IP (no packets are sent) which reveals
        the local interface IP used for internet/LAN traffic.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                # Google's public DNS is commonly reachable; port number is arbitrary
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"

    if not args.ip:
        detected = detect_local_ip()
        print(f"No --ip provided. Auto-detected local IP: {detected}")
        args.ip = detected

    try:
        socket.inet_aton(args.ip)
    except OSError:
        raise SystemExit("Invalid IPv4 address provided: %s" % args.ip)

    zeroconf = Zeroconf(ip_version=IPVersion.V4Only)
    info = build_service_info(args.name, args.ip, args.port)

    try:
        base_name = args.name
        attempt = 0
        while True:
            try:
                display_name = info.server.rstrip('.') if info and hasattr(info, 'server') else base_name
                print(f"Registering A record and service for {display_name}.local -> {args.ip}:{args.port}")
                zeroconf.register_service(info)
                break
            except NonUniqueNameException:
                attempt += 1
                new_name = f"{base_name}-{attempt}"
                print(f"Name conflict: {base_name}.local is already in use. Retrying with {new_name}.local")
                info = build_service_info(new_name, args.ip, args.port)

        # Additionally register a simple address record by registering a ServiceInfo with no type
        # Zeroconf does not provide a direct API to register A records alone, but many clients will
        # learn the host name from the ServiceInfo.server field above. If you need a separate
        # mDNS A record with no service, consider using low-level mDNS packets or another tool.

        print("Advertisement active. Press Ctrl-C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping advertisement...")
    finally:
        try:
            zeroconf.unregister_service(info)
        except Exception:
            pass
        zeroconf.close()


if __name__ == "__main__":
    main()
