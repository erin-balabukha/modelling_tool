# -*- coding: utf-8 -*-
"""
Data for a visualization of summary buckets.
"""

#Data for visualization of bucket summary:
def buckets_vis_summary():
    data_train = np.array([[0,1,56,300,56,300,193.5483870967742,244,244,81.33333333333333,0.18666666666666668,0.18666666666666668,3.612903225806452],
                           [1,2,30,300,86,600,387.0967741935484,270,514,85.66666666666667,0.1,0.14333333333333334,2.774193548387097],
                           [2,3,16,300,102,900,580.6451612903226,284,798,88.66666666666667,0.05333333333333334,0.11333333333333333,2.193548387096774],
                           [3,4,17,300,119,1200,774.1935483870968,283,1081,90.08333333333334,0.056666666666666664,0.09916666666666667,1.9193548387096775],
                           [4,5,9,300,128,1500,967.741935483871,291,1372,91.46666666666667,0.03,0.08533333333333333,1.6516129032258065],
                           [5,6,9,300,137,1800,1161.2903225806451,291,1663,92.38888888888889,0.03,0.07611111111111112,1.4731182795698925],
                           [6,7,9,300,146,2100,1354.8387096774195,291,1954,93.04761904761905,0.03,0.06952380952380953,1.3456221198156684],
                           [7,8,6,300,152,2400,1548.3870967741937,294,2248,93.66666666666667,0.02,0.06333333333333334,1.2258064516129032],
                           [8,9,3,300,155,2700,1741.9354838709676,297,2545,94.25925925925925,0.01,0.05740740740740741,1.1111111111111112],
                           [9,10,0,300,155,3000,1935.483870967742,300,2845,94.83333333333334,0.0,0.051666666666666666,1.0]
                           ])

    data_valid = np.array([[0,1,46,300,46,300,205.47945205479454,254,254,84.66666666666667,0.15333333333333332,0.15333333333333332,3.1506849315068495],
                           [1,2,28,300,74,600,410.95890410958907,272,526,87.66666666666667,0.09333333333333334,0.12333333333333334,2.534246575342466],
                           [2,3,17,300,91,900,616.4383561643835,283,809,89.88888888888889,0.056666666666666664,0.10111111111111111,2.077625570776256],
                           [3,4,18,300,109,1200,821.9178082191781,282,1091,90.91666666666667,0.06,0.09083333333333334,1.8664383561643838],
                           [4,5,10,300,119,1500,1027.3972602739725,290,1381,92.06666666666666,0.03333333333333333,0.07933333333333334,1.6301369863013702],
                           [5,6,6,300,125,1800,1232.876712328767,294,1675,93.05555555555556,0.02,0.06944444444444445,1.4269406392694066],
                           [6,7,5,300,130,2100,1438.3561643835617,295,1970,93.80952380952381,0.016666666666666666,0.06190476190476191,1.2720156555772995],
                           [7,8,10,300,140,2400,1643.8356164383563,290,2260,94.16666666666667,0.03333333333333333,0.058333333333333334,1.1986301369863015],
                           [8,9,4,300,144,2700,1849.3150684931506,296,2556,94.66666666666667,0.013333333333333334,0.05333333333333334,1.0958904109589043],
                           [9,10,2,300,146,3000,2054.794520547945,298,2854,95.13333333333334,0.006666666666666667,0.048666666666666664,1.0]
                           ])

    data_test = np.array([[0,1,35,300,35,300,240.0,265,265,88.33333333333333,0.11666666666666667,0.11666666666666667,2.8000000000000003],
                           [1,2,23,300,58,600,480.0,277,542,90.33333333333333,0.07666666666666666,0.09666666666666666,2.3200000000000003],
                           [2,3,16,300,74,900,720.0,284,826,91.77777777777779,0.05333333333333334,0.08222222222222222,1.9733333333333334],
                           [3,4,11,300,85,1200,960.0,289,1115,92.91666666666667,0.03666666666666667,0.07083333333333333,1.7],
                           [4,5,11,300,96,1500,1200.0,289,1404,93.60000000000001,0.03666666666666667,0.064,1.536],
                           [5,6,9,300,105,1800,1440.0,291,1695,94.16666666666667,0.03,0.058333333333333334,1.4000000000000001],
                           [6,7,9,300,114,2100,1680.0,291,1986,94.57142857142857,0.03,0.054285714285714284,1.302857142857143],
                           [7,8,6,300,120,2400,1920.0,294,2280,95.0,0.02,0.05,1.2000000000000002],
                           [8,9,2,300,122,2700,2160.0,298,2578,95.48148148148148,0.006666666666666667,0.04518518518518518,1.0844444444444445],
                           [9,10,3,300,125,3000,2400.0,297,2875,95.83333333333334,0.01,0.041666666666666664,1.0]
                           ])



    columns = ['index','bucket','actual','predicted','cum_act','cum_pred','ratio_true','false_positive','cum_false_positive','ratio_false','bucket_rate','cum_rate','lift_res']

    train_summary_example = pd.DataFrame(data_train, columns = columns)
    valid_summary_example = pd.DataFrame(data_valid, columns = columns)
    test_summary_example = pd.DataFrame(data_test, columns = columns)

    return train_summary_example, valid_summary_example, test_summary_example
