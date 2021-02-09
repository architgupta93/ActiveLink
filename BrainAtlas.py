"""
Access the Rat Brain Atlas over the internet and show any coordinates that we
might be interested in.
"""

from urllib.request import urlopen
from PIL import Image
from PIL.ImageDraw import Draw
import io
import json
import logging

MODULE_IDENTIFIER = "[BrainAtlas] "
IMAGE_TOP = 50
LINE_COLOR = (0, 255, 0)
LINE_WIDTH = 2
# Set the default coordinates to what you would use for hippocampus.
DEFAULT_ML_COORDINATE = 2.7
DEFAULT_AP_COORDINATE = -4.3
DEFAULT_DV_COORDINATE = 1.5
ENCODING_SCHEME = 'utf-8'

def fetchImage(image_url):
    """
    Fetch an image from url
    """

    try:
        url_content = urlopen(image_url)
        image_data = Image.open(io.BytesIO(url_content.read()))
    except Exception as err:
        logging.error(MODULE_IDENTIFIER + "Unable to read image from URL.")
        print(err)
        return

    return image_data

class WebAtlas(object):
    """
    Access interface to fetch BrainAtlas images from the internet and show any
    set of coordinates in Coronal, Sagittal or Horizontal Slices.

    TODO: We might have to add more features to make this useful. Caching
    images and urls locally would be helpful.
    """

    def __init__(self):
        self._url_base = "http://labs.gaidi.ca/rat-brain-atlas/api.php?"
        self.coronal_image = None
        self.sagittal_image = None
        self.horizontal_image = None

    def fetchCoordinatesAndShowImage(self):
        # Open a dialog box to get the correct corrdinates and show the
        # corresponding image
        pass

    def getCoronalImage(self, ml, ap, dv, show=True):
        """
        Return an image of the Coronal slice at this corrdinate and where in
        the image does the specified coordinate lie.
        """
        url_data = self.queryServer(ml, ap, dv)
        self.coronal_image = fetchImage(url_data['coronal']['image_url']).convert('RGB')
        coordinates = (url_data['coronal']['left'], url_data['coronal']['top'])
        if show:
            placement = Draw(self.coronal_image)
            placement.line([(coordinates[0], IMAGE_TOP), coordinates], fill=(255, 0, 0), width=LINE_WIDTH)
            # TODO: Coloring breaks the drawing
            # placement.line([(coordinates[0], IMAGE_TOP), coordinates])
            self.coronal_image.show(title="%2.2fum"%dv)
        return (self.coronal_image, coordinates)

    def getSagittalImage(self, ml, ap, dv, show=True):
        """
        Return an image of the Sagittal slice at this corrdinate and where in
        the image does the specified coordinate lie.
        """
        url_data = self.queryServer(ml, ap, dv)
        self.sagittal_image = fetchImage(url_data['sagittal']['image_url'])
        coordinates = (url_data['sagittal']['left'], url_data['sagittal']['top'])
        if show:
            placement = Draw(self.sagittal_image)
            # placement.line([(coordinates[0], IMAGE_TOP), coordinates], fill=(0, 0, 225), width=LINE_WIDTH)
            placement.line([(coordinates[0], IMAGE_TOP), coordinates], fill=0)
            self.sagittal_image.show()
        return (self.sagittal_image, coordinates)

    def getHorizontalImage(self, ml, ap, dv, show=True):
        """
        Return an image of the Horizontal slice at this corrdinate and where in
        the image does the specified coordinate lie.
        """
        url_data = self.queryServer(ml, ap, dv)
        self.horizontal_image = fetchImage(url_data['horizontal']['image_url'])
        coordinates = (url_data['horizontal']['left'], url_data['horizontal']['top'])
        if show:
            placement = Draw(self.horizontal_image)
            placement.point([coordinates])
            self.horizontal_image.show()
        return (self.horizontal_image, coordinates)

    def queryServer(self, ml=DEFAULT_ML_COORDINATE, ap=DEFAULT_AP_COORDINATE, dv=DEFAULT_DV_COORDINATE):
        """
        Query the rat brain atlas server to get the correct images for the
        prescribed set of coordinates and return the image urls.
        """

        access_url = self._url_base + "ml=%2.1f&ap=%2.1f&dv=%2.1f"%(ml,ap,dv)
        try:
            url_response = urlopen(access_url)
        except Exception as err:
            logging.error(MODULE_IDENTIFIER + "Unable to read access URL. Check coordinates.")
            print(err)
            return

        # Decode the contents of the webpage
        try:
            url_content = url_response.read().decode(ENCODING_SCHEME)
            url_data_dict = json.loads(url_content)
        except Exception as err:
            logging.error(MODULE_IDENTIFIER + "Unable to parse URL content.")
            print(err)
            return

        # print(url_data_dict)
        return url_data_dict
