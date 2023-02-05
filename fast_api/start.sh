#!/bin/bash
neuron-monitor -c ./monitor.conf |python3.7 /opt/aws/neuron/bin/neuron-monitor-prometheus.py --port 9000 &
python myapp.py