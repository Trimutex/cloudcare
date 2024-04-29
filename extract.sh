DATA_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/data"

if ! test -d "$DATA_DIR"; then
    mkdir $DATA_DIR
fi

if ! test -d "$1"; then
    echo "This script looks for the directory containing .bed files"
    echo "usage: extract.sh /PATH/TO/DIR/"
    exit 1
fi

mkdir /tmp/extract

for file in "$1"/*.bed
do
    filename=$(basename -- "$file")
    bedtools getfasta -bed "$file" -fi "/lfs/hg38.fa" -fo /tmp/extract/"$filename"p
    bedtools complement -i "$file" -g /lfs/hg38.fa.fai > /tmp/extract/"$filename"c 
    bedtools getfasta -bed /tmp/extract/"$filename"c -fi "/lfs/hg38.fa" -fo /tmp/extract/"$filename"n
    echo "Finished with a complement file, cleaning..."
    rm /tmp/extract/"$filename"c -f
    grep -v '>' /tmp/extract/"$filename"p | cut -c 1-120 - | awk 'length($0)==120' > $DATA_DIR/"$filename".dnap;
    echo "Finished with a positive file, cleaning..."
    rm /tmp/extract/"$filename"p -f
    grep -v '>' /tmp/extract/"$filename"n | cut -c 1-120 - | awk 'length($0)==120' > $DATA_DIR/"$filename".dnan;
    echo "Finished with a negative file, cleaning..."
    rm /tmp/extract/"$filename"n -f
done

rmdir /tmp/extract
