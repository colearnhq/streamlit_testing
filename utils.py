import string
import numpy as np


def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text


def extract_text(img, reader):

    g, t, c = ['Nothing'], ['Nothing'], [0.0]
    try:
        result = reader.readtext(img, detail = 1)
        g, t, c = extract_bb_text_confidence(result)
    except:
        pass
    if len(c) == 0:
        g, t, c = ['Nothing'], ['Nothing'], [0.0]
    return g, t, c


def split_text_get_image_name(x):
    return x.split('/')[-1]


def list_to_string(x):
    return ' '.join(x)


def bounding_box_sorting(boxes):
    num_boxes = len(boxes)
    # sort from top to bottom and left to right
    sorted_boxes = sorted(boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)
    # print('::::::::::::::::::::::::::testing')
    # check if the next neighgour box x coordinates is greater then the current box x coordinates if not swap them.
    # repeat the swaping process to a threshold iteration and also select the threshold
    threshold_value_y = 10
    for i in range(5):
      for i in range(num_boxes - 1):
          if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < threshold_value_y and (_boxes[i + 1][0][0] < _boxes[i][0][0]):
              tmp = _boxes[i]
              _boxes[i] = _boxes[i + 1]
              _boxes[i + 1] = tmp
    return _boxes

def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, 0, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


def extract_bb_text_confidence(geometry_text_confidence):
    my_dict = dict(zip(np.arange(len(geometry_text_confidence)),[i[0] for i in geometry_text_confidence]))
    sorted_geometry = [list(my_dict.keys())[list(my_dict.values()).index(i)] for i in bounding_box_sorting([i[0] for i in geometry_text_confidence])]
    result = [geometry_text_confidence[i] for i in sorted_geometry]
    g, t, c = [i[0] for i in result], [i[1] for i in result], [i[2] for i in result]
    return [g, t, c]