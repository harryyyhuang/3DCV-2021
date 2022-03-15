

if [ ! -d data/ ]; then
    mkdir data
fi

cd data

curl -L https://www.dropbox.com/sh/lbkfamu8bwnnwxe/AADC-8H0eVjvtWtGqVla_AdNa?dl=1 > data.zip

unzip data.zip 

rm data.zip
