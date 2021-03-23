import numpy as np
import cv2
import event as ev
from sklearn.mixture import GaussianMixture
import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path

MASK = 1
smaller_mask = 1
FILENAME = ""

def downsample_image(img):
  row, col = img.shape[0]//3, img.shape[1]//3
  down_img = np.zeros((row, col, 3))
  stride = 3
  r , c = 0 , 0
  for i in range(0,img.shape[0],stride):
    c = 0
    for j in range(0,img.shape[1], stride):
      down_img[r][c][0] = np.mean(img[i:i+3, j:j+3, 0])
      down_img[r][c][1] = np.mean(img[i:i+3, j:j+3, 1])
      down_img[r][c][2] = np.mean(img[i:i+3, j:j+3, 2])
      c += 1
    r += 1
  return down_img

def Up_sample_image1(img):
  row, col = img.shape[0]*3, img.shape[1]*3
  up_img = np.zeros((row, col, 3))
  stride = 3
  r , c = 0 , 0
  for i in range(img.shape[0]):
    c = 0
    for j in range(img.shape[1]):      
      up_img[r:r+stride, c:c+stride, :] = img[i][j]
      c += stride
    r += stride
  return up_img

def Up_sample_image(img):
  row, col = img.shape[0]*3, img.shape[1]*3
  up_img = np.zeros((row, col))
  stride = 3
  r , c = 0 , 0
  for i in range(img.shape[0]):
    c = 0
    for j in range(img.shape[1]):
      up_img[r:r+stride, c:c+stride] = img[i][j]
      c += stride
    r += stride
  return up_img

class Grab_Cut():

  def __init__(self, gamma=50, n_iters=5, n_neighbours=4):  
    self.GAMMA = gamma
    self.n_iterations = n_iters
    self.n_neighbours = n_neighbours


  def initialize_ALPHA(self):
      c1, r1, breadth, height = self.flags
      r2 = r1 + height
      c2 = c1 + breadth
      alpha = np.zeros((self.img.shape[0], self.img.shape[1]))
      alpha[r1:r1+height+1 , c1:c1+breadth+1] = 1
      return alpha

  def get_foreground(self, alpha):
      return self.img[np.where(alpha==1)]

  def get_background(self, alpha):
      return self.img[np.where(alpha==0)]

  def get_GMM(self, data, n_comp):
      gm = GaussianMixture(n_components=n_comp, covariance_type='full').fit(data)
      return gm

  def calulate_energy(self, gmm):
      energy = np.zeros((self.img.shape[0], self.img.shape[1]))
      for i in range(self.img.shape[0]):
          for j in range(self.img.shape[1]):
              k = gmm.predict( self.img[i][j].reshape(1, -1))[0]
              energy[i][j] += (np.log(np.linalg.det(gmm.covariances_[k])))/2.0
              tmp = gmm.means_[k] - self.img[i][j]   
              inv = np.linalg.inv(gmm.covariances_[k])
              energy[i][j] += 0.5 * np.matmul(np.matmul(tmp.T, inv), tmp)
              energy[i][j] += -np.log(gmm.weights_[k])    
              #break
      return energy

  def check_index(self, r, c, n, m):
      if ( r>=0 and r<n and c>=0 and c<m ):
          return True
      return False

  def get_beta(self, x, y):
    beta = []
    n = 0
    for i in range(self.img.shape[0]):
      for j in range(self.img.shape[1]):
        self.g.add_node((i, j))
        for row, col in zip(x,y):
          R = i + row
          C = j + col
          if (self.check_index(R, C, self.img.shape[0], self.img.shape[1]) == True):
            #B = np.max((img[i][j]-img[R][C])**2)
            B = np.sum((self.img[i][j] - self.img[R][C])** 2)
            beta.append(B)
    beta = np.array(beta)
    BETA = 0.5 * np.mean(beta)  
    return BETA

  def make_image_graph(self, alpha, E_fg, E_bg ):
      self.g = nx.Graph()
      self.g.add_node('s')
      self.g.add_node('t')
      ##------x & Y INDICES OF NEIGHBOURS INITIALIZATION-----##
      if (self.n_neighbours==8):
          x = [-1, -1, 0, 1, 1, 1, 0, -1]
          y = [0, 1, 1, 1, 0, -1, -1, -1]
      elif (self.n_neighbours==4):
          x = [-1, 0, 1, 0]
          y = [0, 1, 0, -1]
      ##---------------------*****---------------------------##

      ##--------------BETA CALCULATION-----------------##           
      BETA = self.get_beta(x,y)
      ##------------------*****-----------------------##

      for i in range(self.img.shape[0]):
          for j in range(self.img.shape[1]):
              
              ##---------- BACKGROUND = 0 -------------
              if (alpha[i][j]==0):
                  self.g.add_edge('s',(i, j), capacity = 0 )
                  self.g.add_edge((i, j),'t', capacity = 1000000000000000000000000000)
              ##---------- FOREGROUND = 1 -------------
              else:
                  self.g.add_edge('s',(i, j), capacity = E_bg[i][j] )
                  self.g.add_edge((i, j),'t', capacity = E_fg[i][j] )
              ## ------ ADDING NEIGHBOUR (4 OR 8) EDGES -------
              for row, col in zip(x,y):
                  R = i + row
                  C = j + col
                  if (self.check_index(R, C, self.img.shape[0], self.img.shape[1]) == True):
                      wt = self.GAMMA * np.sum(np.exp(-BETA * (self.img[i][j]-self.img[R][C])**2))
                      self.g.add_edge((i,j),(R,C), capacity = self.GAMMA * np.exp(-1.0 * BETA * np.sum((self.img[i][j] - self.img[R][C]) ** 2)))
      return self.g

  def get_segmented_img(self, flags):
    return self.roi_img
  
  def strokes_effect(self, img, alpha):
    #print("STROKES INPUT OPERATED...")
    foreground = self.get_foreground( alpha)
    background = self.get_background( alpha)
    fg_gmm = self.get_GMM(foreground, 2)
    bg_gmm = self.get_GMM(background, 2)
    E_fg = self.calulate_energy( fg_gmm)
    E_bg = self.calulate_energy( bg_gmm)
    GRAPH = self.make_image_graph( alpha, E_fg, E_bg)
    cut_value, partition = nx.minimum_cut(GRAPH, 's', 't', flow_func=shortest_augmenting_path )
    reachable, non_reachable = partition
          
    alphas = np.zeros((self.img.shape[0], self.img.shape[1]))
    for px in reachable:
        if px != 's':
            alphas[px[0]][px[1]] = 1
    self.roi_img = np.zeros(self.img.shape)
    for i in range(img.shape[0]):
        for j in range(self.img.shape[1]):
            if alphas[i][j] == 1:
                self.roi_img[i][j] = self.img[i][j]
    self.roi_img = self.roi_img.astype(np.uint8)
    #print("-----------------------------")
    return alpha

  def fit(self, img, IMG, flags):
      global FILENAME
      name = "itr-"
      iterr = 0
      self.img = img
      #GAMMA = 50
      self.flags = flags
      alpha = self.initialize_ALPHA()
      foreground = self.get_foreground( alpha)
      background = self.get_background( alpha)
      while (self.n_iterations > 0):
          fg_gmm = self.get_GMM(foreground, 2)
          bg_gmm = self.get_GMM(background, 2)
          E_fg = self.calulate_energy( fg_gmm)
          E_bg = self.calulate_energy( bg_gmm)
          GRAPH = self.make_image_graph( alpha, E_fg, E_bg)
          cut_value, partition = nx.minimum_cut(GRAPH, 's', 't', flow_func=shortest_augmenting_path )
          reachable, non_reachable = partition
    
          alphas = np.zeros((self.img.shape[0], self.img.shape[1]))
          for px in reachable:
              if px != 's':
                  alphas[px[0]][px[1]] = 1
          #cv2_imshow(alphas*255)
          self.roi_img = np.zeros(self.img.shape)
          for i in range(img.shape[0]):
              for j in range(self.img.shape[1]):
                  if alphas[i][j] == 1:
                      self.roi_img[i][j] = self.img[i][j]
          self.roi_img = self.roi_img.astype(np.uint8)
          savename = '../images/' + FILENAME[0] + '-' + name + str(iterr) + '.jpg'
          iterr += 1
          cv2.imwrite(savename, self.roi_img)
          #cv2.imshow("roi",self.roi_img)
          #print("-----------------------------")
          self.n_iterations -= 1
          
          #break
      
      #up_img = Up_sample_image1(self.roi_img)
      #cv2.imshow("alphas",alphas)
      up_mask = Up_sample_image(alphas)
      cv2.imshow("up_mask",up_mask)
      up_img = np.zeros(IMG.shape)
      
      return alphas, self.roi_img, up_img, up_mask




def run(filename: str, filename_spiltted):
    global FILENAME
    FILENAME = filename_spiltted
    is_box_operated = False

    COLORS = {
    'BLACK' : [0,0,0],
    'RED'   : [0, 0, 255],
    'GREEN' : [0, 255, 0],
    'BLUE'  : [255, 0, 0],
    'WHITE' : [255,255,255]
    }

    DRAW_BG = {'color' : COLORS['RED'], 'val' : 0}
    DRAW_FG = {'color' : COLORS['WHITE'], 'val' : 1}

    FLAGS = {
        'RECT' : (0, 0, 1, 1),
        'DRAW_STROKE': False,         # flag for drawing strokes
        'DRAW_RECT' : False,          # flag for drawing rectangle
        'rect_over' : False,          # flag to check if rectangle is  drawn
        'rect_or_mask' : -1,          # flag for selecting rectangle or stroke mode
        'value' : DRAW_FG,            # drawing strokes initialized to mark foreground
        'foreground-strokes': [],
        'background-strokes': []
    }

    img = cv2.imread(filename)

    #down_img = downsample_image(img)
    img2 = img.copy()                                
    mask = np.zeros(img.shape[:2], dtype = np.uint8) # mask is a binary array with : 0 - background pixels
                                                     #                               1 - foreground pixels 
    output = np.zeros(img.shape, np.uint8)           # output image to be shown

    # Input and segmentation windows
    cv2.namedWindow('Input Image')
    cv2.namedWindow('Segmented output')
    
    EventObj = ev.EventHandler(FLAGS, img, mask, COLORS)
    cv2.setMouseCallback('Input Image', EventObj.handler)
    cv2.moveWindow('Input Image', img.shape[1] + 10, 90)

    while(1):
        global MASK
        global smaller_mask
        img = EventObj.image
        mask = EventObj.mask
        FLAGS = EventObj.flags
        cv2.imshow('Input Image', img)
        k = cv2.waitKey(1)
        # key bindings
        if k == 27:
            # esc to exit
            break
        
        elif k == ord('0'): 
            #print("Strokes for background")
            # Strokes for background
            FLAGS['value'] = DRAW_BG
        
        elif k == ord('1'):
            # FG drawing
            #print("FG drawing")
            FLAGS['value'] = DRAW_FG
        
        elif k == ord('r'):
            # reset everything
            FLAGS['RECT'] = (0, 0, 1, 1)
            FLAGS['DRAW_STROKE'] = False
            FLAGS['DRAW_RECT'] = False
            FLAGS['rect_or_mask'] = -1
            FLAGS['rect_over'] = False
            FLAGS['value'] = DRAW_FG
            img = img2.copy()
            mask = np.zeros(img.shape[:2], dtype = np.uint8) 
            EventObj.image = img
            EventObj.mask = mask
            output = np.zeros(img.shape, np.uint8)
        
        elif k == 97:# and is_box_operated==False: 
            is_box_operated = True
            #print(EventObj.flags)
            
            img1 = cv2.imread('../images/ronaldo.jpg') 
            #print(img1.shape)
            #cv2.imshow("test",img1)
            if ( img1.shape[0]%3 != 0 or img1.shape[1]%3 != 0 ):
              img1 = cv2.resize(img1, (600, 450))
              #img1 = cv2.resize(img1, (444, 360))  # moon
              #img1 = cv2.resize(img1, (285, 399))  # teddy
            #return
            GBC = Grab_Cut(gamma=500, n_iters=1, n_neighbours=8)
            mask, seg_img, up_img, up_mask = GBC.fit(img, img1,  EventObj.flags['RECT'])
            MASK = up_mask
            smaller_mask = mask             
        
        
        if ( k == 98):
          FG_pixels = EventObj.flags['foreground-strokes']
          BG_pixels = EventObj.flags['background-strokes']
          for px in FG_pixels:
            smaller_mask[px[0]][px[1]] = 1
          for px in BG_pixels:
            smaller_mask[px[0]][px[1]] = 1
          final_mask = GBC.strokes_effect(img, smaller_mask)
          cv2.imshow("final_mask",final_mask)

        EventObj.flags = FLAGS
    return MASK
        

if __name__ == '__main__':
    print("Click and drag right mouse button to make box")
    print("Press 'a' to start the segmentation after drawing bounding box")
    print('for making strokes use left mouse button')
    print('0 - Press 0 key b4 marking background strokes ')
    print('1 - Press 1 key b4 marking background strokes ')
    print("Press 'b' to start the segmentation after drawing strokes")
    path = '../images/'
    filename = 'ronaldo.jpg'               # Path to image file
    FILENAME_ = filename.split('.')
    I = cv2.imread(path+filename)
    if ( I.shape[0]%3 != 0 or I.shape[1]%3 != 0 ):
      I = cv2.resize(I, (600, 450))
      #I = cv2.resize(I, (444, 360)) #moon
      #I = cv2.resize(I, (285, 399))  # teddy
    #print(I.shape)
    down_img = downsample_image(I)
    filename2 = 'down_' + filename
    cv2.imwrite(path+filename2, down_img)
    
    mask = run(path+filename2, FILENAME_)
    up_img = np.zeros(I.shape)
    for i in range(I.shape[0]):
      for j in range(I.shape[1]):
          if mask[i][j] == 1:
            up_img[i][j][0] = I[i][j][0]
            up_img[i][j][1] = I[i][j][1]
            up_img[i][j][2] = I[i][j][2]
    up_img = up_img.astype(np.uint8)
    cv2.imshow("up_img",up_img)
    cv2.imwrite(path + 'seg_' + filename, up_img)
    k = cv2.waitKey(1)
    if k == 27:
      cv2.destroyAllWindows()
