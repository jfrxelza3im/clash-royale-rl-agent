from inference_sdk import InferenceHTTPClient
import base64
from PIL import Image
from io import BytesIO

API_URL = "http://localhost:9001"
API_KEY = "obQog4mAaBRuPZZBIoti"

WORKSPACE = "clashroyalbot-z9idj"
WORKFLOW  = "detect-count-and-visualize"
IMG_PATH  = r'B:\Clash Royal Troop Detection Data Set\MLP FRAMES\Debug Frames\Screenshot_2025.09.14_21.27.00.237.png'

client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)

result = client.run_workflow(
    workspace_name=WORKSPACE,
    workflow_id=WORKFLOW,
    images={"image": IMG_PATH}
)


imgbase = result[0]['img output']

img = base64.b64decode(imgbase)
img = Image.open(BytesIO(img))
img.show()







