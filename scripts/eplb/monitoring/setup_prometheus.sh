#!/bin/bash
# Download and run Prometheus + Grafana

cd /home/ZhanqiuHu/code/vllm/scripts/eplb/monitoring

# Prometheus
if [ ! -d "prometheus-2.48.0.linux-amd64" ]; then
    echo "Downloading Prometheus..."
    wget -q https://github.com/prometheus/prometheus/releases/download/v2.48.0/prometheus-2.48.0.linux-amd64.tar.gz
    tar xzf prometheus-2.48.0.linux-amd64.tar.gz
    rm prometheus-2.48.0.linux-amd64.tar.gz
fi

# Grafana
if [ ! -d "grafana-10.2.2" ]; then
    echo "Downloading Grafana..."
    wget -q https://dl.grafana.com/oss/release/grafana-10.2.2.linux-amd64.tar.gz
    tar xzf grafana-10.2.2.linux-amd64.tar.gz
    rm grafana-10.2.2.linux-amd64.tar.gz
fi

echo "Starting Prometheus on :9090..."
./prometheus-2.48.0.linux-amd64/prometheus --config.file=prometheus.yml --web.listen-address=:9090 &

echo "Starting Grafana on :3000..."
cd grafana-10.2.2
./bin/grafana-server &

echo ""
echo "Access via SSH port forwarding from your local machine:"
echo "  ssh -L 9090:localhost:9090 -L 3000:localhost:3000 user@server"
echo ""
echo "Then open:"
echo "  Prometheus: http://localhost:9090"
echo "  Grafana:    http://localhost:3000 (admin/admin)"
