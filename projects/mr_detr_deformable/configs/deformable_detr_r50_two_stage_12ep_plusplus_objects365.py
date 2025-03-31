import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json

# Define the categories for Objects365
OBJECTS365_CATEGORIES = [{'name': 'Person', 'id': 1},
 {'name': 'Sneakers', 'id': 2},
 {'name': 'Chair', 'id': 3},
 {'name': 'Other Shoes', 'id': 4},
 {'name': 'Hat', 'id': 5},
 {'name': 'Car', 'id': 6},
 {'name': 'Lamp', 'id': 7},
 {'name': 'Glasses', 'id': 8},
 {'name': 'Bottle', 'id': 9},
 {'name': 'Desk', 'id': 10},
 {'name': 'Cup', 'id': 11},
 {'name': 'Street Lights', 'id': 12},
 {'name': 'Cabinet/shelf', 'id': 13},
 {'name': 'Handbag/Satchel', 'id': 14},
 {'name': 'Bracelet', 'id': 15},
 {'name': 'Plate', 'id': 16},
 {'name': 'Picture/Frame', 'id': 17},
 {'name': 'Helmet', 'id': 18},
 {'name': 'Book', 'id': 19},
 {'name': 'Gloves', 'id': 20},
 {'name': 'Storage box', 'id': 21},
 {'name': 'Boat', 'id': 22},
 {'name': 'Leather Shoes', 'id': 23},
 {'name': 'Flower', 'id': 24},
 {'name': 'Bench', 'id': 25},
 {'name': 'Potted Plant', 'id': 26},
 {'name': 'Bowl/Basin', 'id': 27},
 {'name': 'Flag', 'id': 28},
 {'name': 'Pillow', 'id': 29},
 {'name': 'Boots', 'id': 30},
 {'name': 'Vase', 'id': 31},
 {'name': 'Microphone', 'id': 32},
 {'name': 'Necklace', 'id': 33},
 {'name': 'Ring', 'id': 34},
 {'name': 'SUV', 'id': 35},
 {'name': 'Wine Glass', 'id': 36},
 {'name': 'Belt', 'id': 37},
 {'name': 'Moniter/TV', 'id': 38},
 {'name': 'Backpack', 'id': 39},
 {'name': 'Umbrella', 'id': 40},
 {'name': 'Traffic Light', 'id': 41},
 {'name': 'Speaker', 'id': 42},
 {'name': 'Watch', 'id': 43},
 {'name': 'Tie', 'id': 44},
 {'name': 'Trash bin Can', 'id': 45},
 {'name': 'Slippers', 'id': 46},
 {'name': 'Bicycle', 'id': 47},
 {'name': 'Stool', 'id': 48},
 {'name': 'Barrel/bucket', 'id': 49},
 {'name': 'Van', 'id': 50},
 {'name': 'Couch', 'id': 51},
 {'name': 'Sandals', 'id': 52},
 {'name': 'Bakset', 'id': 53},
 {'name': 'Drum', 'id': 54},
 {'name': 'Pen/Pencil', 'id': 55},
 {'name': 'Bus', 'id': 56},
 {'name': 'Wild Bird', 'id': 57},
 {'name': 'High Heels', 'id': 58},
 {'name': 'Motorcycle', 'id': 59},
 {'name': 'Guitar', 'id': 60},
 {'name': 'Carpet', 'id': 61},
 {'name': 'Cell Phone', 'id': 62},
 {'name': 'Bread', 'id': 63},
 {'name': 'Camera', 'id': 64},
 {'name': 'Canned', 'id': 65},
 {'name': 'Truck', 'id': 66},
 {'name': 'Traffic cone', 'id': 67},
 {'name': 'Cymbal', 'id': 68},
 {'name': 'Lifesaver', 'id': 69},
 {'name': 'Towel', 'id': 70},
 {'name': 'Stuffed Toy', 'id': 71},
 {'name': 'Candle', 'id': 72},
 {'name': 'Sailboat', 'id': 73},
 {'name': 'Laptop', 'id': 74},
 {'name': 'Awning', 'id': 75},
 {'name': 'Bed', 'id': 76},
 {'name': 'Faucet', 'id': 77},
 {'name': 'Tent', 'id': 78},
 {'name': 'Horse', 'id': 79},
 {'name': 'Mirror', 'id': 80},
 {'name': 'Power outlet', 'id': 81},
 {'name': 'Sink', 'id': 82},
 {'name': 'Apple', 'id': 83},
 {'name': 'Air Conditioner', 'id': 84},
 {'name': 'Knife', 'id': 85},
 {'name': 'Hockey Stick', 'id': 86},
 {'name': 'Paddle', 'id': 87},
 {'name': 'Pickup Truck', 'id': 88},
 {'name': 'Fork', 'id': 89},
 {'name': 'Traffic Sign', 'id': 90},
 {'name': 'Ballon', 'id': 91},
 {'name': 'Tripod', 'id': 92},
 {'name': 'Dog', 'id': 93},
 {'name': 'Spoon', 'id': 94},
 {'name': 'Clock', 'id': 95},
 {'name': 'Pot', 'id': 96},
 {'name': 'Cow', 'id': 97},
 {'name': 'Cake', 'id': 98},
 {'name': 'Dinning Table', 'id': 99},
 {'name': 'Sheep', 'id': 100},
 {'name': 'Hanger', 'id': 101},
 {'name': 'Blackboard/Whiteboard', 'id': 102},
 {'name': 'Napkin', 'id': 103},
 {'name': 'Other Fish', 'id': 104},
 {'name': 'Orange/Tangerine', 'id': 105},
 {'name': 'Toiletry', 'id': 106},
 {'name': 'Keyboard', 'id': 107},
 {'name': 'Tomato', 'id': 108},
 {'name': 'Lantern', 'id': 109},
 {'name': 'Machinery Vehicle', 'id': 110},
 {'name': 'Fan', 'id': 111},
 {'name': 'Green Vegetables', 'id': 112},
 {'name': 'Banana', 'id': 113},
 {'name': 'Baseball Glove', 'id': 114},
 {'name': 'Airplane', 'id': 115},
 {'name': 'Mouse', 'id': 116},
 {'name': 'Train', 'id': 117},
 {'name': 'Pumpkin', 'id': 118},
 {'name': 'Soccer', 'id': 119},
 {'name': 'Skiboard', 'id': 120},
 {'name': 'Luggage', 'id': 121},
 {'name': 'Nightstand', 'id': 122},
 {'name': 'Tea pot', 'id': 123},
 {'name': 'Telephone', 'id': 124},
 {'name': 'Trolley', 'id': 125},
 {'name': 'Head Phone', 'id': 126},
 {'name': 'Sports Car', 'id': 127},
 {'name': 'Stop Sign', 'id': 128},
 {'name': 'Dessert', 'id': 129},
 {'name': 'Scooter', 'id': 130},
 {'name': 'Stroller', 'id': 131},
 {'name': 'Crane', 'id': 132},
 {'name': 'Remote', 'id': 133},
 {'name': 'Refrigerator', 'id': 134},
 {'name': 'Oven', 'id': 135},
 {'name': 'Lemon', 'id': 136},
 {'name': 'Duck', 'id': 137},
 {'name': 'Baseball Bat', 'id': 138},
 {'name': 'Surveillance Camera', 'id': 139},
 {'name': 'Cat', 'id': 140},
 {'name': 'Jug', 'id': 141},
 {'name': 'Broccoli', 'id': 142},
 {'name': 'Piano', 'id': 143},
 {'name': 'Pizza', 'id': 144},
 {'name': 'Elephant', 'id': 145},
 {'name': 'Skateboard', 'id': 146},
 {'name': 'Surfboard', 'id': 147},
 {'name': 'Gun', 'id': 148},
 {'name': 'Skating and Skiing shoes', 'id': 149},
 {'name': 'Gas stove', 'id': 150},
 {'name': 'Donut', 'id': 151},
 {'name': 'Bow Tie', 'id': 152},
 {'name': 'Carrot', 'id': 153},
 {'name': 'Toilet', 'id': 154},
 {'name': 'Kite', 'id': 155},
 {'name': 'Strawberry', 'id': 156},
 {'name': 'Other Balls', 'id': 157},
 {'name': 'Shovel', 'id': 158},
 {'name': 'Pepper', 'id': 159},
 {'name': 'Computer Box', 'id': 160},
 {'name': 'Toilet Paper', 'id': 161},
 {'name': 'Cleaning Products', 'id': 162},
 {'name': 'Chopsticks', 'id': 163},
 {'name': 'Microwave', 'id': 164},
 {'name': 'Pigeon', 'id': 165},
 {'name': 'Baseball', 'id': 166},
 {'name': 'Cutting/chopping Board', 'id': 167},
 {'name': 'Coffee Table', 'id': 168},
 {'name': 'Side Table', 'id': 169},
 {'name': 'Scissors', 'id': 170},
 {'name': 'Marker', 'id': 171},
 {'name': 'Pie', 'id': 172},
 {'name': 'Ladder', 'id': 173},
 {'name': 'Snowboard', 'id': 174},
 {'name': 'Cookies', 'id': 175},
 {'name': 'Radiator', 'id': 176},
 {'name': 'Fire Hydrant', 'id': 177},
 {'name': 'Basketball', 'id': 178},
 {'name': 'Zebra', 'id': 179},
 {'name': 'Grape', 'id': 180},
 {'name': 'Giraffe', 'id': 181},
 {'name': 'Potato', 'id': 182},
 {'name': 'Sausage', 'id': 183},
 {'name': 'Tricycle', 'id': 184},
 {'name': 'Violin', 'id': 185},
 {'name': 'Egg', 'id': 186},
 {'name': 'Fire Extinguisher', 'id': 187},
 {'name': 'Candy', 'id': 188},
 {'name': 'Fire Truck', 'id': 189},
 {'name': 'Billards', 'id': 190},
 {'name': 'Converter', 'id': 191},
 {'name': 'Bathtub', 'id': 192},
 {'name': 'Wheelchair', 'id': 193},
 {'name': 'Golf Club', 'id': 194},
 {'name': 'Briefcase', 'id': 195},
 {'name': 'Cucumber', 'id': 196},
 {'name': 'Cigar/Cigarette ', 'id': 197},
 {'name': 'Paint Brush', 'id': 198},
 {'name': 'Pear', 'id': 199},
 {'name': 'Heavy Truck', 'id': 200},
 {'name': 'Hamburger', 'id': 201},
 {'name': 'Extractor', 'id': 202},
 {'name': 'Extention Cord', 'id': 203},
 {'name': 'Tong', 'id': 204},
 {'name': 'Tennis Racket', 'id': 205},
 {'name': 'Folder', 'id': 206},
 {'name': 'American Football', 'id': 207},
 {'name': 'earphone', 'id': 208},
 {'name': 'Mask', 'id': 209},
 {'name': 'Kettle', 'id': 210},
 {'name': 'Tennis', 'id': 211},
 {'name': 'Ship', 'id': 212},
 {'name': 'Swing', 'id': 213},
 {'name': 'Coffee Machine', 'id': 214},
 {'name': 'Slide', 'id': 215},
 {'name': 'Carriage', 'id': 216},
 {'name': 'Onion', 'id': 217},
 {'name': 'Green beans', 'id': 218},
 {'name': 'Projector', 'id': 219},
 {'name': 'Frisbee', 'id': 220},
 {'name': 'Washing Machine/Drying Machine', 'id': 221},
 {'name': 'Chicken', 'id': 222},
 {'name': 'Printer', 'id': 223},
 {'name': 'Watermelon', 'id': 224},
 {'name': 'Saxophone', 'id': 225},
 {'name': 'Tissue', 'id': 226},
 {'name': 'Toothbrush', 'id': 227},
 {'name': 'Ice cream', 'id': 228},
 {'name': 'Hotair ballon', 'id': 229},
 {'name': 'Cello', 'id': 230},
 {'name': 'French Fries', 'id': 231},
 {'name': 'Scale', 'id': 232},
 {'name': 'Trophy', 'id': 233},
 {'name': 'Cabbage', 'id': 234},
 {'name': 'Hot dog', 'id': 235},
 {'name': 'Blender', 'id': 236},
 {'name': 'Peach', 'id': 237},
 {'name': 'Rice', 'id': 238},
 {'name': 'Wallet/Purse', 'id': 239},
 {'name': 'Volleyball', 'id': 240},
 {'name': 'Deer', 'id': 241},
 {'name': 'Goose', 'id': 242},
 {'name': 'Tape', 'id': 243},
 {'name': 'Tablet', 'id': 244},
 {'name': 'Cosmetics', 'id': 245},
 {'name': 'Trumpet', 'id': 246},
 {'name': 'Pineapple', 'id': 247},
 {'name': 'Golf Ball', 'id': 248},
 {'name': 'Ambulance', 'id': 249},
 {'name': 'Parking meter', 'id': 250},
 {'name': 'Mango', 'id': 251},
 {'name': 'Key', 'id': 252},
 {'name': 'Hurdle', 'id': 253},
 {'name': 'Fishing Rod', 'id': 254},
 {'name': 'Medal', 'id': 255},
 {'name': 'Flute', 'id': 256},
 {'name': 'Brush', 'id': 257},
 {'name': 'Penguin', 'id': 258},
 {'name': 'Megaphone', 'id': 259},
 {'name': 'Corn', 'id': 260},
 {'name': 'Lettuce', 'id': 261},
 {'name': 'Garlic', 'id': 262},
 {'name': 'Swan', 'id': 263},
 {'name': 'Helicopter', 'id': 264},
 {'name': 'Green Onion', 'id': 265},
 {'name': 'Sandwich', 'id': 266},
 {'name': 'Nuts', 'id': 267},
 {'name': 'Speed Limit Sign', 'id': 268},
 {'name': 'Induction Cooker', 'id': 269},
 {'name': 'Broom', 'id': 270},
 {'name': 'Trombone', 'id': 271},
 {'name': 'Plum', 'id': 272},
 {'name': 'Rickshaw', 'id': 273},
 {'name': 'Goldfish', 'id': 274},
 {'name': 'Kiwi fruit', 'id': 275},
 {'name': 'Router/modem', 'id': 276},
 {'name': 'Poker Card', 'id': 277},
 {'name': 'Toaster', 'id': 278},
 {'name': 'Shrimp', 'id': 279},
 {'name': 'Sushi', 'id': 280},
 {'name': 'Cheese', 'id': 281},
 {'name': 'Notepaper', 'id': 282},
 {'name': 'Cherry', 'id': 283},
 {'name': 'Pliers', 'id': 284},
 {'name': 'CD', 'id': 285},
 {'name': 'Pasta', 'id': 286},
 {'name': 'Hammer', 'id': 287},
 {'name': 'Cue', 'id': 288},
 {'name': 'Avocado', 'id': 289},
 {'name': 'Hamimelon', 'id': 290},
 {'name': 'Flask', 'id': 291},
 {'name': 'Mushroon', 'id': 292},
 {'name': 'Screwdriver', 'id': 293},
 {'name': 'Soap', 'id': 294},
 {'name': 'Recorder', 'id': 295},
 {'name': 'Bear', 'id': 296},
 {'name': 'Eggplant', 'id': 297},
 {'name': 'Board Eraser', 'id': 298},
 {'name': 'Coconut', 'id': 299},
 {'name': 'Tape Measur/ Ruler', 'id': 300},
 {'name': 'Pig', 'id': 301},
 {'name': 'Showerhead', 'id': 302},
 {'name': 'Globe', 'id': 303},
 {'name': 'Chips', 'id': 304},
 {'name': 'Steak', 'id': 305},
 {'name': 'Crosswalk Sign', 'id': 306},
 {'name': 'Stapler', 'id': 307},
 {'name': 'Campel', 'id': 308},
 {'name': 'Formula 1 ', 'id': 309},
 {'name': 'Pomegranate', 'id': 310},
 {'name': 'Dishwasher', 'id': 311},
 {'name': 'Crab', 'id': 312},
 {'name': 'Hoverboard', 'id': 313},
 {'name': 'Meat ball', 'id': 314},
 {'name': 'Rice Cooker', 'id': 315},
 {'name': 'Tuba', 'id': 316},
 {'name': 'Calculator', 'id': 317},
 {'name': 'Papaya', 'id': 318},
 {'name': 'Antelope', 'id': 319},
 {'name': 'Parrot', 'id': 320},
 {'name': 'Seal', 'id': 321},
 {'name': 'Buttefly', 'id': 322},
 {'name': 'Dumbbell', 'id': 323},
 {'name': 'Donkey', 'id': 324},
 {'name': 'Lion', 'id': 325},
 {'name': 'Urinal', 'id': 326},
 {'name': 'Dolphin', 'id': 327},
 {'name': 'Electric Drill', 'id': 328},
 {'name': 'Hair Dryer', 'id': 329},
 {'name': 'Egg tart', 'id': 330},
 {'name': 'Jellyfish', 'id': 331},
 {'name': 'Treadmill', 'id': 332},
 {'name': 'Lighter', 'id': 333},
 {'name': 'Grapefruit', 'id': 334},
 {'name': 'Game board', 'id': 335},
 {'name': 'Mop', 'id': 336},
 {'name': 'Radish', 'id': 337},
 {'name': 'Baozi', 'id': 338},
 {'name': 'Target', 'id': 339},
 {'name': 'French', 'id': 340},
 {'name': 'Spring Rolls', 'id': 341},
 {'name': 'Monkey', 'id': 342},
 {'name': 'Rabbit', 'id': 343},
 {'name': 'Pencil Case', 'id': 344},
 {'name': 'Yak', 'id': 345},
 {'name': 'Red Cabbage', 'id': 346},
 {'name': 'Binoculars', 'id': 347},
 {'name': 'Asparagus', 'id': 348},
 {'name': 'Barbell', 'id': 349},
 {'name': 'Scallop', 'id': 350},
 {'name': 'Noddles', 'id': 351},
 {'name': 'Comb', 'id': 352},
 {'name': 'Dumpling', 'id': 353},
 {'name': 'Oyster', 'id': 354},
 {'name': 'Table Teniis paddle', 'id': 355},
 {'name': 'Cosmetics Brush/Eyeliner Pencil', 'id': 356},
 {'name': 'Chainsaw', 'id': 357},
 {'name': 'Eraser', 'id': 358},
 {'name': 'Lobster', 'id': 359},
 {'name': 'Durian', 'id': 360},
 {'name': 'Okra', 'id': 361},
 {'name': 'Lipstick', 'id': 362},
 {'name': 'Cosmetics Mirror', 'id': 363},
 {'name': 'Curling', 'id': 364},
 {'name': 'Table Tennis ', 'id': 365}]


def get_objects365_metadata():
    thing_ids = [k["id"] for k in OBJECTS365_CATEGORIES]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": [k["name"] for k in OBJECTS365_CATEGORIES],
    }



from .deformable_detr_r50_12ep import train, dataloader, optimizer, model

# modify model config
model.with_box_refine = True
model.as_two_stage = True

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/deformable_detr_r50_two_stage_12ep"

model.transformer.encoder.feedforward_dim=2048
model.transformer.decoder.feedforward_dim=2048

model.transformer.encoder.attn_dropout=0.0
model.transformer.encoder.ffn_dropout=0.0
model.transformer.decoder.attn_dropout=0.0
model.transformer.decoder.ffn_dropout=0.0

model.mixed_selection = True
model.transformer.mixed_selection = True
model.transformer.decoder.look_forward_twice = True


model.select_box_nums_for_evaluation = 300
model.num_queries = 900
model.num_classes = 365
model.criterion.num_classes = 365
train.eval_period = 100000

# model.transformer.encoder.use_checkpoint=True

dataloader.train.num_workers=12

train_json_path = "Objects365/train/zhiyuan_objv2_train_new.json"
val_json_path = "Objects365/val/zhiyuan_objv2_val.json"
objects365_root = "/Objects365/"

from detectron2.data.datasets import register_coco_instances

# Register the training dataset
register_coco_instances(
    "objects365_train", 
    get_objects365_metadata(), 
    train_json_path, 
    objects365_root + "train/"
)
register_coco_instances(
    "objects365_val", 
    get_objects365_metadata(), 
    val_json_path, 
    objects365_root + "val/"
)

dataloader.train.dataset.names = "objects365_train"
dataloader.test.dataset.names = "objects365_val"

train.max_iter = int(1728778 * 4 / 16 - 1)


############### schedule ###############

from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler

def custom_objects365_pretrain_scheduler(epochs=26, decay_epochs=24, warmup_epochs=0, batch_size=16):
    """
    Returns the config for a default multi-step LR scheduler such as "50epochs",
    commonly referred to in papers, where every 1x has the total length of 1440k
    training images (~12 COCO epochs). LR is decayed once at the end of training.

    Args:
        epochs (int): total training epochs.
        decay_epochs (int): lr decay steps.
        warmup_epochs (int): warmup epochs.

    Returns:
        DictConfig: configs that define the multiplier for LR during training
    """
    # 1728778
    total_steps = epochs * 1728778 // batch_size    
    decay_steps = decay_epochs * 1728778 // batch_size
    warmup_steps = warmup_epochs * 1728778 // batch_size
    scheduler = L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[decay_steps, total_steps],
    )
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_steps / total_steps,
        warmup_method="linear",
        warmup_factor=0.001,
    )

lr_multiplier = custom_objects365_pretrain_scheduler(epochs=4, decay_epochs=3, warmup_epochs=0, batch_size=16)