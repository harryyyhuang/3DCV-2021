

if [ ! -d frames/ ]; then
    mkdir frames
fi

cd frames

curl -L https://www.dropbox.com/sh/6kfrog5qtiqlpg7/AADxivcD_DW20prK0CQXX5oda?dl=1 > frames.zip

unzip frames.zip 

rm frames.zip
