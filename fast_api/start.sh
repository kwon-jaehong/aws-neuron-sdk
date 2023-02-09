#!/bin/bash
exec neuron-monitor -c ./monitor.conf |python3 ./t_test.py --port 9000
# exec python3 myapp.py &
# exec neuron-monitor -c ./monitor.conf |python3 ./mod-neuron-monitor-prometheus.py --port 9000
