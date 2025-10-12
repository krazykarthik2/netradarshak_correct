## Expose this machine as `specsserver.local`

This repo includes two ways to make the IP address reachable as `specsserver.local`:

1. Local-only: add an entry into the Windows hosts file. This affects only the current machine (or any machine you edit the hosts file on).
2. LAN-wide (recommended): run the included Python mDNS announcer to publish `specsserver.local` via mDNS/Bonjour. Other devices with mDNS support (macOS, Linux with avahi, Windows with Bonjour installed) will resolve the name automatically.

Files added:
- `announce_mdns.py` — Python script that uses zeroconf to advertise `specsserver.local` on the LAN.
- `requirements.txt` — contains the Python dependency `zeroconf`.

Quick start — hosts file (local-only)

1. Open Notepad as Administrator and edit `C:\Windows\System32\drivers\etc\hosts`.
2. Add a line below (replace the IP with your machine's LAN IP):

   192.168.1.42 specsserver.local

3. Save. You should now be able to reach the server at `http://specsserver.local` (from the same machine).

Quick start — mDNS (LAN-wide)

1. Create and activate a Python virtual environment (recommended).
2. Install dependencies:

   pip install -r requirements.txt

3. Run the announcer (replace the IP and port as needed):

   python announce_mdns.py --ip 192.168.1.42 --port 80

The script runs until you Ctrl-C it. While it's running it will advertise a service and the A-record for `specsserver.local` on the local network via mDNS.

Notes and troubleshooting

- mDNS requires the client machines to support Bonjour/Avahi. macOS and most Linux distros do by default; Windows often needs Apple's Bonjour or third-party mDNS resolver.
- Hosts-file edits are immediate but local to each machine you edit.
- If you want a permanent LAN-wide DNS name without depending on mDNS, configure a static DNS entry on your router or run a small DNS server (e.g. dnsmasq) and point clients to it.

If you want, I can also:
- Add a small HTTP server example that binds to the same IP/port and can be discovered.
- Show router-specific steps for common consumer routers if you tell me the model.

Running the included Flask server so the service resolves without a port

The repository includes a small Flask server in `server.py` that exposes `/caption` and `/ocr` endpoints. By default it now attempts to bind to port 80 so the service is reachable at `http://specsserver.local` (no `:5000` required).

On Windows you normally need Administrator privileges to bind to port 80. Examples (cmd.exe):

Run as Administrator and start on port 80:

   python server.py --port 80

Run without elevation (will fall back to port 5000 if port 80 is unavailable or you lack permissions):

   python server.py

If you prefer a different port, pass `--port <PORT>`.

If you run the Flask server on port 80 and also run `announce_mdns.py --ip <your-ip> --port 80`, clients on the LAN that support mDNS will be able to access the service at `http://specsserver.local`.
