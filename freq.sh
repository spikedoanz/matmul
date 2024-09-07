#!/bin/bash

while true; do
    date +"%H:%M:%S"
    cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq
    echo "-----"
    sleep 1
done
