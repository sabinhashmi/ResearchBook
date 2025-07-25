AlgorithmPath1="/afs/cern.ch/work/s/skalavan/FileHub/DownstreamTrackingAlgorithm/WorkingPoint/Moore/Pr/PrAlgorithms/src/PrHybridSeeding.cpp"
AlgorithmPath2="/afs/cern.ch/work/s/skalavan/FileHub/DownstreamTrackingAlgorithm/WorkingPoint/Moore/Pr/PrAlgorithms/src/PrLongLivedTracking.cpp"


OptionFile="/afs/cern.ch/work/s/skalavan/FileHub/DownstreamTrackingAlgorithm/Scripts/DownstreamOptionsv2.py"
WorkingMoore='/afs/cern.ch/work/s/skalavan/FileHub/DownstreamTrackingAlgorithm/WorkingPoint/Moore'
RunMoore="/afs/cern.ch/work/s/skalavan/FileHub/DownstreamTrackingAlgorithm/WorkingPoint/Moore/run"
Logs="/afs/cern.ch/work/s/skalavan/FileHub/DownstreamTrackingAlgorithm/WorkingPoint/Data/Logs"

CSV_Output="ThresholdOutput.csv"


mkdir -p "$Logs"

Thresholds1=(0.0 0.1 0.2 0.3 0.4 0.5 0.6)
Thresholds2=(0.0 0.1 0.2 0.3 0.4 0.5 0.6)

for threshold1 in "${Thresholds1[@]}"
do

    for threshold2 in "${Thresholds2[@]}"

    do
        echo "Processing HybridSeedingThreshold: $threshold1 and LonglivedTrackingThreshold: $threshold2"
        sed -i -e "s/if (probability < .*) continue;$/if (probability < $threshold1) continue;/" $AlgorithmPath1
        sed -i -e "s/if (probability < .*) continue;$/if (probability < $threshold2) continue;/" $AlgorithmPath2

        echo "CompilingMoore..."
        if ! make -C $WorkingMoore; then
        echo "Make failed!"
        exit 1
        fi

        echo "Running the script and producing outputs!"
        # $RunMoore gaudirun.py $OptionFile | tee $Logs/log_$threshold1_$threshold2.log
        $RunMoore gaudirun.py $OptionFile | tee $Logs/log_$(echo "$threshold1" | sed 's/\./_/')_$(echo "$threshold2" | sed 's/\./_/').log



        echo "File Generated Successfully!"
    done
done


echo "HybridSeedingThreshold,LonglivedThreshold,TotalSeedTracks,SeedGhostTracks,TotalDownstreamTracks,DownstreamGhostTracks" > "$Logs/$CSV_Output"

for logfile in "$Logs"/*.log;
do
    if [ -e "$logfile" ]; then
        # threshold1=$(echo "$logfile" | sed -E 's/.*log_([0-9]+(?:_[0-9]+)?)_([0-9]+(?:_[0-9]+)?)\.log/\1/')
        threshold1=$(echo "$logfile" | sed -E 's/.*log_([0-9]+_[0-9]+)_([0-9]+_[0-9]+)\.log/\1/' | sed 's/_/./')

        # threshold2=$(echo "$logfile" | sed -E 's/.*log_([0-9]+(?:_[0-9]+)?)_([0-9]+(?:_[0-9]+)?)\.log/\2/')
        threshold2=$(echo "$logfile" | sed -E 's/.*log_([0-9]+_[0-9]+)_([0-9]+_[0-9]+)\.log/\2/' | sed 's/_/./')


        downstream=$(grep -oP "INFO\s+\*\*\*\* Downstream\s+\K\d+" "$logfile") 
        downstream_ghost=$(grep -oP "tracks\s+including\s+\K\d+" "$logfile" | head -n 1)

        seed=$(grep -oP "INFO\s+\*\*\*\* Seed\s+\K\d+" "$logfile") 
        seed_ghost=$(grep -oP "tracks\s+including\s+\K\d+" "$logfile" | tail -n 1)  

        
        
        echo "$threshold1,$threshold2,$seed,$seed_ghost,$downstream,$downstream_ghost" >> "$Logs/$CSV_Output"
    else
        echo "Error: Log file not found - $logfile"
    fi
done

echo "Combined Log File Generated: $Logs/$CSV_Output"