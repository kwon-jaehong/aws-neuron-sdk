#!/bin/bash
exec python3 myapp.py
exec neuron-monitor -c ./monitor.conf |python3 ./mod-neuron-monitor-prometheus.py --port 9000
