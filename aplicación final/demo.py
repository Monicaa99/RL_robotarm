"""
Demo of HMR.

Note that HMR requires the bounding box of the person in the image. The best performance is obtained when max length of the person in the image is roughly 150px.

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out the bbox and the right scale factor.

Sample usage:

# On images on a tightly cropped image around the person
python -m demo --img_path data/im1963.jpg
python -m demo --img_path data/coco1.png

# On images, with openpose output
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting
from scipy.spatial.transform import Rotation as R

import sys
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel

from PIL import Image
import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))

import darknet as dn
import pdb
import cv2
import time

global model_load
model_load=False
global sess
global model

def applyTransformationMatrix(T, listOfPoints):

	points = []
	for p in listOfPoints:
		p = np.array([p[0],p[1],p[2],1])
		p_ = np.dot(T,p.T).tolist()[0:3]
		points.append(p_)
	return np.array(points)



def computeNormalizationT(origen, soporte1, soporte2):

	A = soporte1 - origen
	B = soporte2 - origen
	C = np.cross(A,B) # producto vectorial para obtener el vector normal al plano que forman A,B
	# -- En la v2 del programa se modifica el orden del cross para poder producir una matriz con det=1
	# -- En la version original la instruccion era: B = np.cross(A,C). Daba det=-1
	B = np.cross(C,A) # producto vectorial para obtener el vector normal al plano que forman A,B

	# se convierten los vectores en unitarios
	A = A / np.linalg.norm(A)
	B = B / np.linalg.norm(B)
	C = C / np.linalg.norm(C)

	# origen del nuevo sistema
	originBody = np.hstack((origen,[1]))

	# el origen del sistema antiguo esta en el leap
	originCamera = np.array([0,0,0,1])

	aprim = [1,0,0]
	bprim = [0,1,0]
	cprim = [0,0,1]

	row1 = [A[0], B[0], C[0]]
	row2 = [A[1], B[1], C[1]]
	row3 = [A[2], B[2], C[2]]

	Aexp = np.hstack((row1,[originBody[0]]))
	Bexp = np.hstack((row2,[originBody[1]]))
	Cexp = np.hstack((row3,[originBody[2]]))

	Coeficiente = np.array([row1, row2, row3])
	CoeficienteExp = np.array([Aexp,Bexp,Cexp,[0,0,0,1]])

	vector1 = np.linalg.solve(Coeficiente,aprim)
	vector2 = np.linalg.solve(Coeficiente,bprim)
	vector3 = np.linalg.solve(Coeficiente,cprim)
	vector4 = np.linalg.solve(CoeficienteExp,originCamera)

	extraRow = np.array([0,0,0,1])

	translationMatrix = np.array([vector4[0],vector4[1],vector4[2]])
	rotationMatrix = np.array([vector1,vector2,vector3]).T
	rotationMatrix = np.hstack((rotationMatrix, translationMatrix.reshape(-1,1)))

	transformationMatrix = np.vstack((rotationMatrix, extraRow.reshape(1,-1)))

	return transformationMatrix




flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_string(
	'json_path', None,
	'If specified, uses the openpose output to crop the image.')


def visualize(img, proc_param, joints, verts, cam):
	"""
	Renders the result in original image coordinate frame.
	"""
	cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
		proc_param, verts, cam, joints, img_size=img.shape[:2])

	# Render results
	skel_img = vis_util.draw_skeleton(img, joints_orig)
	rend_img_overlay = renderer(
		vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
	rend_img = renderer(
		vert_shifted, cam=cam_for_render, img_size=img.shape[:2])
	rend_img_vp1 = renderer.rotated(
		vert_shifted, 60, cam=cam_for_render, img_size=img.shape[:2])
	rend_img_vp2 = renderer.rotated(
		vert_shifted, -60, cam=cam_for_render, img_size=img.shape[:2])

	import matplotlib.pyplot as plt
	# plt.ion()
	plt.figure(1)
	plt.clf()
	plt.subplot(231)
	plt.imshow(img)
	plt.title('input')
	plt.axis('off')
	plt.subplot(232)
	plt.imshow(skel_img)
	plt.title('joint projection')
	plt.axis('off')
	plt.subplot(233)
	plt.imshow(rend_img_overlay)
	plt.title('3D Mesh overlay')
	plt.axis('off')
	plt.subplot(234)
	plt.imshow(rend_img)
	plt.title('3D mesh')
	plt.axis('off')
	plt.subplot(235)
	plt.imshow(rend_img_vp1)
	plt.title('diff vp')
	plt.axis('off')
	plt.subplot(236)
	plt.imshow(rend_img_vp2)
	plt.title('diff vp')
	plt.axis('off')
	plt.draw()
	plt.show()
	# import ipdb
	# ipdb.set_trace()


def preprocess_image(img_path, json_path=None):
	img = io.imread(img_path)
	if img.shape[2] == 4:
		img = img[:, :, :3]

	if json_path is None:
		if np.max(img.shape[:2]) != config.img_size:
			print('Resizing so the max image size is %d..' % config.img_size)
			scale = (float(config.img_size) / np.max(img.shape[:2]))
		else:
			scale = 1.
		center = np.round(np.array(img.shape[:2]) / 2).astype(int)
		# image center in (x,y)
		center = center[::-1]
	else:
		scale, center = op_util.get_bbox(json_path)

	crop, proc_param = img_util.scale_and_crop(img, scale, center,
											   config.img_size)

	# Normalize image to [-1, 1]
	crop = 2 * ((crop / 255.) - 0.5)

	return crop, proc_param, img


def main(img_path, json_path=None):
	global model_load
	global sess
	global model
	if model_load==False:
		sess = tf.Session()

		model = RunModel(config, sess=sess)
		model_load=True
	#model = RunModel(config)

	print(img_path)
	input_img, proc_param, img = preprocess_image(img_path, json_path)
	# Add batch dimension: 1 x D x D x 3
	input_img = np.expand_dims(input_img, 0)

	# Theta is the 85D vector holding [camera, pose, shape]
	# where camera is 3D [s, tx, ty]
	# pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
	# shape is 10D shape coefficients of SMPL
	joints, verts, cams, joints3d, theta = model.predict(
		input_img, get_theta=True)

	#visualize(img, proc_param, joints[0], verts[0], cams[0])

	joints3d = joints3d[0]
	print(joints3d)


	origin = joints3d[9] # lshoulder
	soporte1 = joints3d[12] # neck
	soporte2 = joints3d[3] # lhip

	print(origin)
	print(soporte1)
	print(soporte2)

	T = computeNormalizationT(origin, soporte1, soporte2)
	print("T", T)
	joints3d = applyTransformationMatrix(T, joints3d)

	r = R.from_euler('X', -90, degrees=True)
	joints3d = r.apply(joints3d)
	r = R.from_euler('Y',-20, degrees=True)
	joints3d = r.apply(joints3d)
	r = R.from_euler('X', 15, degrees=True)
	joints3d = r.apply(joints3d)
	r = R.from_euler('Z', 15, degrees=True)
	joints3d = r.apply(joints3d)
	r = R.from_euler('Z', -90, degrees=True)
	joints3d = r.apply(joints3d)


	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')

	xs = joints3d[:,0]
	ys = joints3d[:,1]
	zs = joints3d[:,2]

	ax.scatter(xs, ys, zs, marker="o")

	ax.scatter([0], [0], [0], marker="*")

	ax.plot([joints3d[12,0], joints3d[13,0]],
			[joints3d[12,1], joints3d[13,1]],
			[joints3d[12,2], joints3d[13,2]],color = 'g') # head 13 neck 12
	ax.plot([joints3d[8,0], joints3d[12,0]],
			[joints3d[8,1], joints3d[12,1]],
			[joints3d[8,2], joints3d[12,2]],color = 'g') # rshoulder 8 neck 12
	ax.plot([joints3d[9,0], joints3d[12,0]],
			[joints3d[9,1], joints3d[12,1]],
			[joints3d[9,2], joints3d[12,2]],color = 'g') # lshoulder 9 neck 12
	ax.plot([joints3d[8,0], joints3d[7,0]],
			[joints3d[8,1], joints3d[7,1]],
			[joints3d[8,2], joints3d[7,2]],color = 'g') # rshoulder 8 relbow 7
	ax.plot([joints3d[6,0], joints3d[7,0]],
			[joints3d[6,1], joints3d[7,1]],
			[joints3d[6,2], joints3d[7,2]],color = 'b') # rwrist 6 relbow 7
	ax.plot([joints3d[9,0], joints3d[10,0]],
			[joints3d[9,1], joints3d[10,1]],
			[joints3d[9,2], joints3d[10,2]],color = 'g') # lshoulder 9 lelbow 10
	ax.plot([joints3d[11,0], joints3d[10,0]],
			[joints3d[11,1], joints3d[10,1]],
			[joints3d[11,2], joints3d[10,2]],color = 'r') # lwrist 11 lelbow 10
	ax.plot([joints3d[8,0], joints3d[2,0]],
			[joints3d[8,1], joints3d[2,1]],
			[joints3d[8,2], joints3d[2,2]],color = 'g') # rshoulder 8 rhip 2
	ax.plot([joints3d[9,0], joints3d[3,0]],
			[joints3d[9,1], joints3d[3,1]],
			[joints3d[9,2], joints3d[3,2]],color = 'g') # lshoulder 9 lhip 3
	ax.plot([joints3d[2,0], joints3d[3,0]],
			[joints3d[2,1], joints3d[3,1]],
			[joints3d[2,2], joints3d[3,2]],color = 'g') # rhip 2 lhip 3
	ax.plot([joints3d[2,0], joints3d[1,0]],
			[joints3d[2,1], joints3d[1,1]],
			[joints3d[2,2], joints3d[1,2]],color = 'g') # rhip 2 rknee 1
	ax.plot([joints3d[0,0], joints3d[1,0]],
			[joints3d[0,1], joints3d[1,1]],
			[joints3d[0,2], joints3d[1,2]],color = 'g') # rankle 0 rknee 1
	ax.plot([joints3d[3,0], joints3d[4,0]],
			[joints3d[3,1], joints3d[4,1]],
			[joints3d[3,2], joints3d[4,2]],color = 'g') # lhip 3 lknee 4
	ax.plot([joints3d[5,0], joints3d[4,0]],
			[joints3d[5,1], joints3d[4,1]],
			[joints3d[5,2], joints3d[4,2]],color = 'g') # lankle 5 lknee 4

	d=[joints3d[11,0]-joints3d[10,0],joints3d[11,1]-joints3d[10,1],joints3d[11,2]-joints3d[10,2]]
	d=np.linalg.norm(d)
	print(d)
	ratio=150/d
	print(ratio)
	goal=[joints3d[11,0]*ratio,joints3d[11,1]*ratio,joints3d[11,2]*ratio]
	print("goal: ",goal)


	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	ax.set_xlim([-1, 1])
	ax.set_ylim([-1, 1])
	ax.set_zlim([-1, 1])

	#plt.show()
	return goal


if __name__ == '__main__':

	dn.set_gpu(0)
	net = dn.load_net("/home/monicapina/TFG/yolo/hmr/darknet/cfg/yolov3.cfg", "/home/monicapina/TFG/yolo/hmr/darknet/yolov3.weights", 0)
	meta = dn.load_meta("/home/monicapina/TFG/yolo/hmr/cfg/coco.data")


	model_load=False
	while(True):

		file_exist=False
		while file_exist==False:
			print("Waiting image...")
			time.sleep(5)
			if os.path.exists("/home/monicapina/TFG/yolo/hmr/images_pepper/image.jpg") is True:
				file_exist=True

		r = dn.detect(net, meta,"/home/monicapina/TFG/yolo/hmr/images_pepper/image.jpg") #Lee imagen del robot
		time.sleep(3)
		img = Image.open("/home/monicapina/TFG/yolo/hmr/images_pepper/image.jpg")
		person=r[0]
		coordenadas=person[2]
		img_res = img.crop((coordenadas[0]-coordenadas[2], coordenadas[1]-coordenadas[3], coordenadas[0]+coordenadas[2], coordenadas[1]+coordenadas[3]))

		img_res.save("/home/monicapina/TFG/yolo/hmr/images_pepper/yolo_result.jpg")
		img.close()
		config = flags.FLAGS
		config(sys.argv)
		# Using pre-trained model, change this to use your own.
		config.load_path = src.config.PRETRAINED_MODEL

		config.batch_size = 1

		renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

		goal=main("/home/monicapina/TFG/yolo/hmr/images_pepper/yolo_result.jpg")

		if os.path.exists("/home/monicapina/TFG/yolo/hmr/images_pepper/yolo_result.jpg") is True:

			os.remove("/home/monicapina/TFG/yolo/hmr/images_pepper/yolo_result.jpg")

		if os.path.exists("/home/monicapina/TFG/yolo/hmr/images_pepper/goal.txt") is True:

			os.remove("/home/monicapina/TFG/yolo/hmr/images_pepper/goal.txt")


		file = open('/home/monicapina/TFG/yolo/hmr/images_pepper/goal.txt', 'w')
		file.write(str(goal[0])+" "+str(goal[1])+" "+str(goal[2]))
		file.close()



	#Borrar imagen al procesar
