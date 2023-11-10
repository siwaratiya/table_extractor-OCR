# views.py
import cv2
import layoutparser as lp
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import io
@csrf_exempt
def extract_table(request):
    if request.method == 'POST':
        # Get the image from the POST request
        image_file = request.FILES.get('image')

        # Read the image file and extract the text from it
        image = Image.open(io.BytesIO(image_file.read()))
        model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.65],label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
        layout = model.detect(image)
        text_blocks = lp.Layout([b for b in layout if b.type=="Table"])
        figure_blocks = lp.Layout([b for b in layout if b.type=='Figure'])
        text_blocks = lp.Layout([b for b in text_blocks \
                   if not any(b.is_in(b_fig) for b_fig in figure_blocks)])
        # Extract the table from the image text
        # table_extractor = TableExtractor()
        tables = []

        # Return the extracted table(s) as JSON
        return JsonResponse({'tables': tables})

    # Return an error message if the request method is not POST
    return JsonResponse({'error': 'Invalid request method'})
