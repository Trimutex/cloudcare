mkdir /tmp/extract
for f in *.bed
do
    bedtools getfasta -bed "$f" -fi "../hg38.fa" -fo /tmp/extract/"$f"p.txt
    bedtools complement -i "$f" -g ../hg38.fa.fai > /tmp/extract/"$f"n 
    bedtools getfasta -bed /tmp/extract/"$f"n -fi "../hg38.fa" -fo /tmp/extract/"$f"n.txt
done

for f in /tmp/extract/*.p.txt
do
    do grep -v '>' "$f" | cut -c 1-120 - | awk 'length($0)==120' > "$f".dnap;
done

for f in /tmp/extract/*.n.txt
do
    do grep -v '>' "$f" | cut -c 1-120 - | awk 'length($0)==120' > "$f".dnan;
done
