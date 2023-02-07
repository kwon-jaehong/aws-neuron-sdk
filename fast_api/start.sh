#!/bin/bash
neuron-monitor -c ./monitor.conf |python3.7 ./mod-neuron-monitor-prometheus.py --port 9000 &
python myapp.py