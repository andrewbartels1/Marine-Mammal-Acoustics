# pull the sample flac
cd /app/to_process

echo "pulling 3 sample files to process from the google bucket of raw audio data"
gsutil -m cp \
  "gs://noaa-passive-bioacoustic/adeon/chb/amar.493.4/audio/AMAR493.4.20180624T121936Z.flac" \
  "gs://noaa-passive-bioacoustic/adeon/chb/amar.493.4/audio/AMAR493.4.20180624T123936Z.flac" \
  "gs://noaa-passive-bioacoustic/adeon/chb/amar.493.4/audio/AMAR493.4.20180624T125936Z.flac" \
  .
# pull the meta data!
gsutil -m cp -r \
  "gs://noaa-passive-bioacoustic/adeon/chb/amar.493.4/metadata" \
  .
cd ../

echo "running analysis now!"

python audio_processing/DataLoader.py demo

echo "Data finished processing!"

echo "check the 'to_process' folder for any generated outputs! NOTE: if metdata wasn't found, the samples aren't inserted into the database"
