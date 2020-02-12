
#
import matplotlib.pyplot as plt


preview_path = '/home/dcm-admin/cnn-train-test-3-party/yolov3-pdnetv1/output'

f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(plt.imread(preview_path + '/3.jpg'))
axarr[0,1].imshow(plt.imread(preview_path + '/4.jpg'))
axarr[1,0].imshow(plt.imread(preview_path + '/5.jpg'))
axarr[1,1].imshow(plt.imread(preview_path + '/8.jpg'))

plt.show()