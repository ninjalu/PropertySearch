import requests
from PIL import Image
import json
from io import BytesIO
import base64

img = Image.open('...')
buffer = BytesIO()
img.save(buffer, format='JPG')
img_bytes = base64.b64decode(buffer.getvalue())
img_str = img_bytes.decode('utf-8')

s = {
    'input': img_str
}
r = requests.post('http://0.0.0.0:5000/predict', data=json.dumps(s))


print(r)
