import json


class Sensor:
    def __init__(self, path):
        file = open(path)
        self._metadata = json.load(file)

        if len(self._metadata) == 28:
            self.name = self._metadata["DATA_COLLECTION_NAME"]
            self.long = self._metadata["DEPLOY_LON"]
            self.lat = self._metadata["DEPLOY_LAT"]
        else:
            self.name = self._metadata["site_aliases"][0]
            self.long = self._metadata['deployment']['lon']
            self.lat = self._metadata['deployment']['lat']
            

    def location(self):
        return (self.long, self.lat)
