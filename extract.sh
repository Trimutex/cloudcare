mkdir /tmp/extract
for f in *.bed
do
    bedtools getfasta -bed "$f" -fi "/lfs/hg38.fa" -fo /tmp/extract/"$f"p
    bedtools complement -i "$f" -g /lfs/hg38.fa.fai > /tmp/extract/"$f"c 
    bedtools getfasta -bed /tmp/extract/"$f"c -fi "/lfs/hg38.fa" -fo /tmp/extract/"$f"n
    echo "Finished with a complement file, cleaning..."
    rm /tmp/extract/"$f"c -f
    grep -v '>' /tmp/extract/"$f"p | cut -c 1-120 - | awk 'length($0)==120' > /tmp/extract/"$f".dnap;
    echo "Finished with a positive file, cleaning..."
    rm /tmp/extract/"$f"p -f
    grep -v '>' /tmp/extract/"$f"n | cut -c 1-120 - | awk 'length($0)==120' > /tmp/extract/"$f".dnan;
    echo "Finished with a negative file, cleaning..."
    rm /tmp/extract/"$f"n -f
done
