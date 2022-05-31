
ANN_ID:int = 0
AUTHOR_ID:int = 0
IMG_ID:int = 1
BOX_IDX:int = 2
SEG_IDX:int = 3
MIN_BOX_SIZE:int = 25

INK_THICKNESS = [1,2,3]

INK_COLORS = [ ('Black', (0, 0, 0)), 
               ('Dark gray', (68, 68, 68)), 
               ('Light gray', (204,204,204)), 
               ('White', (255, 255, 255)),
               ('Blue', (0,0,255)), 
               ('Red', (255, 0, 0)), 
               ('Green', (45,201,55))
            ]

MAPPING = {
           'text': [['Textblock', 'Textline', 'List', 'Table', 'Diagram', 'Drawing'], [0.25, 0.25, 0.2,0.1,0.1,0.1]], 
           'title' : [['Word', 'Textline'], [0.5,0.5]], 
           'list' : [['List', 'Textblock'], [0.5, 0.5]], 
           'table' : [['Table', 'Textblock'],[.8, 0.2]],
           'figure' : [['Diagram', 'Drawing', 'Textblock'],[0.4,0.4,0.2]]
           }

#Background: 0
BASE_CLASSES = [
           'Machine-Table', #1 
           'Machine-Figure', #2
           'Machine-List', #3
           'Machine-Textline', #4 
           'Machine-Textblock', #5
           'Handwritten-Table', #6
           'Handwritten-Figure', #7
           'Handwritten-List', #8
           'Handwritten-Textblock', #9
           'Handwritten-Textline', #10
           ] 
MARKUPS = ['Marking_Bracket', 'Marking_Encircling', 'Marking_Underline', 'Arrow']

ALL_CLASSES = BASE_CLASSES + MARKUPS
CLASS_DICT = { k+1:v for k,v in enumerate(ALL_CLASSES) } 

DENSE_LAYERS_ = [[1,2,6,7],[3,4,5,8,9,10],[11,12,13,14]]
DENSE_LAYERS = dict( (v,k) for k,a in enumerate(DENSE_LAYERS_) for v in a )


DISTINCT_COLORS = [ 
                    "xkcd:black",
                    "xkcd:purple",
                    "xkcd:green",
                    "xkcd:blue",
                    "xkcd:brown",
                    "xkcd:yellow",
                    "xkcd:red",
                    "xkcd:orange",
                    "xkcd:teal", 
                    "xkcd:light green",
                    "xkcd:light purple",
                    "xkcd:greenish grey",
                    "xkcd:navy",
                    "xkcd:gold",
                    "xkcd:raspberry",
                   ]
