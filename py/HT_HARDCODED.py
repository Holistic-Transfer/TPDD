import torch


# OfficeHome
OFFICEHOME_SOURCE_AR = torch.tensor(
    [2, 58, 46, 21, 19, 34, 7, 56, 39, 30, 22, 61, 33, 20, 59, 53, 51, 6, 16, 27, 24, 44, 28, 1, 37, 41, 12, 36, 40, 63,
     0, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15, 17, 18, 23, 25, 26, 29, 31, 32, 35, 38, 42, 43, 45, 47, 48, 49, 50, 52, 54,
     55, 57, 60, 62, 64])
OFFICEHOME_SOURCE_RW = torch.tensor(
    [14, 50, 42, 38, 64, 48, 13, 63, 15, 24, 55, 19, 60, 3, 16, 41, 62, 46, 2, 49, 4, 11, 25, 51, 32, 30, 31, 6, 18, 59,
     39, 17, 20, 12, 56, 23, 10, 27, 61, 1, 29, 8, 37, 28, 26, 45, 34, 52, 35, 36, 33, 57, 43, 21, 0, 9, 7, 47, 54, 44,
     40, 5, 22, 53, 58])
OFFICEHOME_N_CLASSES = 65


def GET_VISIBLE_CLASSES(dataset, source, target, n_seen_classes):
    if dataset == 'OfficeHome':
        if source == 'Ar':
            clzes = OFFICEHOME_SOURCE_AR
        elif source == 'Rw':
            clzes = OFFICEHOME_SOURCE_RW
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    return clzes, clzes[:n_seen_classes]


# # ImageNet
# IMAGENER_TARGET_R = torch.tensor(
#     [341, 245, 613, 319, 9, 355, 583, 13, 327, 330, 161, 441, 71, 472, 515, 199, 23, 487, 701, 163, 22, 288, 437, 250,
#      338, 570, 824, 147, 365, 462, 448, 965, 717, 26, 637, 457, 178, 105, 866, 130, 476, 951, 311, 546, 84, 281, 335,
#      269, 172, 254, 63, 309, 347, 852, 895, 260, 97, 31, 162, 555, 301, 231, 981, 483, 94, 435, 148, 390, 100, 219, 259,
#      155, 145, 310, 232, 430, 947, 948, 208, 957, 195, 596, 99, 413, 820, 953, 401, 594, 361, 8, 96, 79, 113, 132, 47,
#      954, 1, 768, 393, 251, 933, 812, 323, 932, 308, 367, 414, 314, 988, 340, 293, 76, 447, 593, 724, 234, 763, 160,
#      470, 943, 334, 337, 247, 125, 907, 587, 889, 621, 629, 150, 945, 815, 90, 6, 425, 299, 407, 368, 847, 107, 388,
#      617, 936, 931, 11, 428, 658, 171, 558, 983, 937, 315, 833, 242, 276, 29, 779, 928, 980, 263, 787, 203, 372, 397,
#      292, 187, 4, 344, 883, 296, 780, 471, 289, 949, 366, 235, 776, 2, 151, 277, 774, 657, 579, 122, 267, 469, 362, 265,
#      39, 934, 144, 875, 463, 805, 963, 353, 967, 291, 609, 207])
# IMAGENET_TARGET_S = torch.tensor(
#     [312, 897, 632, 147, 309, 106, 599, 711, 763, 223, 972, 595, 819, 780, 463, 598, 78, 518, 126, 948, 787, 553, 193,
#      415, 187, 548, 416, 536, 737, 853, 336, 105, 181, 714, 586, 764, 963, 512, 420, 929, 774, 633, 502, 123, 629, 159,
#      94, 551, 544, 756, 773, 915, 974, 176, 726, 646, 969, 562, 932, 11, 895, 666, 526, 438, 75, 732, 473, 840, 758,
#      547, 626, 943, 330, 131, 527, 754, 292, 578, 911, 768, 609, 921, 6, 778, 528, 380, 625, 236, 317, 251, 654, 628,
#      581, 601, 920, 760, 991, 993, 418, 241, 108, 996, 185, 219, 719, 511, 888, 110, 81, 580, 820, 291, 539, 523, 923,
#      224, 381, 671, 573, 797, 204, 151, 552, 792, 3, 248, 707, 664, 197, 218, 556, 76, 831, 200, 959, 36, 529, 635, 423,
#      850, 138, 940, 608, 879, 254, 906, 606, 670, 496, 910, 226, 467, 855, 992, 239, 967, 247, 443, 982, 785, 278, 475,
#      642, 31, 484, 543, 370, 687, 451, 922, 79, 570, 45, 999, 256, 859, 825, 313, 49, 736, 650, 320, 321, 211, 450, 152,
#      712, 678, 383, 286, 458, 925, 741, 68, 583, 735, 604, 190, 144, 29, 298, 258, 771, 334, 755, 269, 454, 136, 393,
#      916, 363, 968, 183, 966, 80, 500, 550, 16, 373, 743, 103, 215, 698, 871, 387, 882, 288, 343, 175, 338, 863, 857,
#      386, 801, 716, 649, 125, 717, 833, 405, 961, 610, 65, 725, 655, 287, 703, 872, 47, 432, 23, 58, 813, 437, 294, 585,
#      284, 192, 837, 540, 229, 896, 209, 349, 509, 257, 9, 225, 727, 870, 811, 264, 407, 329, 90, 508, 70, 238, 409, 384,
#      72, 781, 549, 148, 377, 883, 816, 689, 479, 944, 682, 139, 440, 829, 559, 328, 731, 157, 358, 15, 99, 441, 57, 927,
#      67, 680, 487, 360, 8, 169, 233, 524, 480, 541, 66, 74, 772, 89, 624, 88, 817, 121, 130, 406, 851, 462, 40, 722,
#      427, 935, 44, 314, 425, 482, 533, 636, 516, 289, 674, 615, 589, 307, 293, 43, 730, 908, 442, 876, 83, 802, 342, 97,
#      985, 881, 958, 656, 955, 844, 954, 436, 505, 331, 132, 46, 973, 210, 492, 685, 728, 385, 691, 498, 266, 198, 191,
#      325, 280, 164, 140, 356, 699, 748, 267, 100, 53, 740, 311, 259, 745, 994, 404, 391, 826, 794, 483, 160, 640, 5,
#      255, 48, 39, 901, 19, 350, 398, 426, 951, 460, 62, 823, 348, 989, 783, 206, 50, 913, 962, 261, 212, 414, 4, 987,
#      61, 981, 117, 893, 439, 832, 471, 73, 706, 998, 574, 300, 54, 322, 84, 55, 648, 858, 563, 216, 866, 353, 234, 203,
#      346, 575, 424, 96, 91, 102, 643, 975, 12, 362, 713, 165, 909, 246, 134, 316, 885, 864, 240, 285, 593, 983, 252,
#      596, 476, 953, 445, 878, 242, 95, 410, 179, 30, 411, 873, 742, 310, 894, 515, 156, 856, 877, 862, 13, 525, 846,
#      493, 827, 607, 303, 659, 477, 395, 613, 788, 135, 124, 812, 765, 178, 786, 262, 301, 32, 449, 459, 340, 357, 237,
#      376, 214, 394, 600, 397, 815, 684, 803, 42, 245, 347, 345, 769, 842, 964, 934, 17, 971, 918, 366, 616, 87, 489,
#      898, 949, 814, 250, 137, 970, 572, 565, 752, 638, 115, 865, 744, 644, 930, 611, 369, 355, 641, 592, 579, 665, 302,
#      928, 361, 434, 988, 990, 21, 571, 154, 235, 413, 519, 486, 658, 818, 753, 145, 770, 835, 92, 290, 472, 847, 20,
#      677, 627, 133, 952, 447, 408, 839, 662, 784, 618, 180, 995, 128, 936, 466, 683, 351, 561, 697, 457, 631, 41, 926,
#      194, 522, 614, 304, 672, 738, 950, 0, 335, 465, 777, 208, 688, 762, 51, 221, 978, 554, 799, 119, 485, 503, 374, 26,
#      163, 354, 957, 270, 588, 382, 534, 767, 274, 56, 806, 63, 173, 495, 146, 367, 751, 542, 766, 676, 941, 597, 327,
#      35, 868, 757, 273, 98, 577, 902, 867, 207, 344, 116, 521, 947, 917, 228, 584, 700, 800, 530, 723, 52, 127, 702,
#      651, 59, 177, 481, 339, 222, 686, 861, 444, 513, 260, 497, 696, 749, 724, 174, 149, 669, 205, 37, 171, 822, 341,
#      390, 845, 166, 775, 693, 271, 841, 914, 694, 227, 34, 630, 268, 793, 933, 332, 612, 161, 155, 182, 109, 118, 491,
#      653, 28, 834, 912, 804, 230, 887, 924, 701, 557, 217, 69, 249, 150, 2, 419, 71, 645, 808, 504, 33, 283, 705, 545,
#      809, 401, 843, 931, 446, 431, 708, 567, 692, 306, 747, 202, 469, 277, 85, 396, 622, 652, 824, 709, 977, 582, 564,
#      890, 141, 195, 905, 560, 1, 299, 402, 111, 520, 143, 379, 10, 244, 456, 499, 886, 265, 253, 412, 279, 400, 875,
#      805, 372, 18, 761, 537, 750, 452, 243, 602, 591, 637, 488, 657, 371, 546, 403, 501, 296, 796, 690, 903, 295, 576,
#      27, 904, 729, 170, 960, 510, 122, 621, 639, 673, 704, 880, 517, 789, 464, 663, 199, 852, 667, 433, 720, 874, 718,
#      282, 112, 25, 590, 836, 555, 399, 113, 474, 470, 782, 681, 120, 620, 568, 388, 566, 184, 14, 733, 860, 494, 679,
#      531, 421, 490, 830, 668, 201, 453, 231, 326, 790, 734, 776, 795, 319, 7, 538, 324, 359, 168, 93, 710, 569, 429,
#      984, 364, 759, 945, 634, 647, 828, 810, 189, 695, 186, 869, 378, 375, 276, 623, 675, 937, 821, 889, 430, 153, 392,
#      798, 435, 891, 478, 965, 318, 979, 907, 746, 281, 661, 196, 514, 272, 507, 417, 315, 365, 939, 82, 38, 142, 660,
#      308, 899, 938, 532, 77, 448, 535, 107, 172, 976, 779, 980, 919, 455, 900, 297, 461, 389, 849, 587, 232, 884, 594,
#      158, 352, 114, 64, 368, 956, 104, 848, 263, 838, 213, 558, 892, 739, 60, 807, 986, 946, 942, 506, 24, 619, 323,
#      305, 428, 791, 603, 167, 188, 86, 422, 101, 715, 468, 22, 617, 337, 129, 997, 605, 333, 162, 220, 721, 854, 275])
# IMAGENET_N_CLASSES = 1000
#
# # vtab
# VTAB_TARGET_CALTECH101 = np.array(
#     [67, 48, 10, 91, 77, 24, 62, 40, 20, 59, 86, 78, 88, 14, 79, 39, 89, 50, 18, 3, 95, 87, 49, 85, 81, 21, 43, 73, 63,
#      27, 2, 37, 11, 84, 61, 46, 6, 60, 82, 31, 100, 93, 64, 56, 76, 72, 51, 69, 1, 32, 65, 42, 8, 71, 12, 16, 83, 38,
#      96, 54, 74, 52, 36, 5, 94, 66, 44, 53, 68, 97, 57, 19, 90, 17, 92, 41, 58, 101, 0, 70, 22, 4, 55, 9, 47, 29, 34,
#      25, 15, 33, 99, 98, 80, 13, 75, 23, 26, 7, 45, 28, 35, 30])
# VTAB_TARGET_CIFAR100 = np.array(
#     [47, 71, 60, 65, 15, 46, 31, 23, 18, 79, 34, 20, 7, 77, 62, 42, 12, 40, 2, 84, 78, 88, 85, 75, 4, 91, 83, 82, 76,
#      95, 63, 80, 51, 53, 13, 64, 29, 93, 81, 38, 59, 56, 69, 6, 5, 48, 1, 35, 41, 49, 52, 74, 9, 57, 58, 28, 30, 27, 68,
#      66, 3, 90, 36, 45, 73, 11, 0, 39, 43, 25, 32, 37, 97, 10, 98, 19, 16, 70, 50, 67, 14, 92, 24, 22, 55, 96, 94, 72,
#      17, 86, 33, 54, 89, 87, 8, 21, 26, 99, 44, 61])
# VTAB_TARGET_DTD = np.array(
#     [13, 33, 41, 19, 5, 38, 44, 35, 11, 37, 0, 30, 17, 46, 1, 43, 39, 28, 32, 12, 23, 36, 40, 3, 22, 14, 15, 20, 18, 6,
#      29, 45, 9, 25, 26, 4, 27, 21, 42, 24, 34, 10, 16, 8, 2, 7, 31])
# VTAB_TARGET_EUROSAT = np.array([3, 8, 7, 2, 1, 6, 0, 9, 4, 5])
# VTAB_TARGET_FLOWER102 = np.array(
#     [70, 29, 40, 28, 80, 49, 33, 42, 67, 69, 65, 53, 44, 20, 56, 94, 0, 36, 100, 74, 72, 90, 27, 18, 10, 14, 59, 101,
#      52, 31, 51, 92, 6, 2, 24, 64, 88, 5, 50, 7, 19, 79, 91, 93, 38, 58, 84, 34, 71, 62, 60, 54, 87, 9, 39, 26, 86, 78,
#      37, 12, 73, 63, 98, 68, 82, 75, 76, 30, 43, 15, 16, 95, 66, 4, 99, 45, 17, 3, 8, 23, 96, 21, 41, 25, 1, 13, 55, 85,
#      97, 89, 48, 11, 61, 47, 35, 46, 57, 81, 22, 77, 83, 32])
# VTAB_TARGET_PETS = np.array(
#     [34, 28, 3, 19, 23, 9, 26, 32, 21, 20, 25, 11, 16, 1, 27, 7, 14, 4, 17, 29, 24, 36, 18, 12, 0, 8, 31, 33, 13, 10,
#      15, 30, 5, 2, 22, 35, 6])
# VTAB_TARGET_RESISC45 = np.array(
#     [11, 33, 41, 31, 5, 37, 19, 9, 30, 21, 22, 39, 7, 35, 13, 23, 3, 28, 0, 44, 24, 43, 6, 18, 32, 29, 34, 25, 12, 17,
#      1, 8, 27, 42, 15, 4, 2, 16, 36, 10, 38, 14, 20, 40, 26])
# VTAB_TARGET_SUN397 = np.array(
#     [210, 25, 15, 340, 200, 303, 35, 247, 206, 192, 235, 262, 388, 203, 100, 350, 224, 228, 358, 258, 32, 97, 112, 51,
#      349, 205, 289, 384, 104, 202, 106, 364, 342, 374, 343, 52, 155, 309, 261, 283, 266, 217, 24, 131, 99, 91, 178, 251,
#      60, 327, 243, 294, 232, 50, 355, 172, 357, 179, 269, 300, 156, 53, 275, 154, 324, 103, 122, 45, 353, 264, 77, 189,
#      368, 138, 191, 238, 361, 147, 315, 114, 152, 270, 280, 308, 339, 96, 157, 177, 175, 373, 173, 276, 263, 26, 209,
#      185, 63, 284, 30, 229, 221, 92, 166, 64, 312, 125, 95, 256, 313, 120, 220, 320, 348, 385, 117, 211, 184, 329, 314,
#      146, 136, 227, 295, 13, 233, 180, 338, 359, 285, 366, 396, 76, 151, 115, 328, 169, 130, 393, 73, 78, 391, 386, 277,
#      390, 230, 68, 241, 90, 257, 161, 23, 352, 57, 265, 22, 326, 231, 347, 272, 394, 168, 174, 380, 222, 41, 33, 239,
#      307, 0, 323, 124, 121, 282, 305, 48, 244, 171, 362, 129, 331, 70, 110, 20, 310, 39, 346, 144, 365, 176, 134, 383,
#      290, 259, 82, 148, 204, 395, 274, 382, 293, 322, 54, 139, 311, 142, 109, 187, 296, 27, 196, 141, 292, 260, 149,
#      181, 186, 212, 197, 249, 74, 344, 126, 89, 101, 49, 8, 213, 193, 93, 43, 190, 218, 252, 225, 240, 208, 360, 236,
#      375, 55, 116, 133, 111, 381, 65, 188, 85, 335, 288, 61, 132, 369, 67, 143, 17, 145, 299, 158, 378, 332, 298, 246,
#      7, 271, 160, 387, 86, 42, 137, 164, 317, 268, 334, 279, 119, 153, 165, 38, 150, 318, 44, 250, 83, 321, 29, 297,
#      273, 306, 113, 330, 88, 128, 234, 370, 135, 278, 351, 183, 255, 242, 226, 163, 379, 11, 4, 56, 253, 123, 291, 345,
#      46, 94, 87, 10, 207, 356, 37, 107, 66, 214, 6, 301, 354, 5, 372, 286, 194, 319, 223, 28, 34, 40, 316, 219, 254, 58,
#      9, 75, 84, 36, 267, 16, 199, 325, 118, 59, 237, 376, 69, 47, 81, 281, 341, 159, 245, 108, 140, 105, 62, 79, 1, 2,
#      12, 162, 392, 216, 371, 337, 170, 72, 31, 302, 336, 3, 18, 14, 389, 80, 98, 304, 167, 201, 198, 333, 363, 248, 287,
#      71, 21, 215, 367, 195, 127, 377, 19, 182, 102])