netsh advfirewall firewall add rule name="Flask 80" protocol=TCP localport=80 action=allow dir=in
start cmd /c python announce_mdns.py &
python server.py