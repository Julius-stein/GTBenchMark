#/bin/bash
echo "===== CPU ====="
lscpu

echo "===== Memory ====="
free -h
dmidecode -t memory | egrep "Size|Speed" | uniq

echo "===== GPU ====="
nvidia-smi -L
nvidia-smi topo -m
nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version --format=csv

echo "===== Disk ====="
lsblk -d -o NAME,MODEL,SIZE,ROTA
df -hT | grep -E "^/dev"

echo "===== Network ====="
lspci | grep -i ether
ip -brief link show

echo "===== System ====="
uname -a
cat /etc/os-release | egrep "PRETTY_NAME|VERSION="
