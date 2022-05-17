from deeptexturestf import DeepTexture
from multiprocessing import Process
from PIL import Image
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

feature_layers = ['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 'block5_pool','var_loss']
name = 'tex'
losses = []

finalLosses = {}
instanceList = []
scoreList    = []
processList  = []

def getMinIndex():
    if len(losses)<1:
        return
    index = 0
    min_ = losses[index]
    for i in range(len(losses)):
        temp = losses[i]
        print("these are the tensors: ",temp)
        if (min_ > temp):
            index = i
            min_ = temp
    return index


def createLoss():
    
    for i in feature_layers:
        min_ = 999999999999999999999999999999
        mintensor = None
        for j in range(len(instanceList)):
            if (instanceList[j].layer_loss_scores[i] < min_):
                min_ = instanceList[j].layer_loss_scores[i]
                mintensor = j

        if (mintensor == None):
            raise ValueError("Outstandingly large value for all losses, check texture image")
        finalLosses[i] = mintensor

    return finalLosses


def initializeList(base_path,tex_path_list):
    for i in range(len(tex_path_list)):
        currentName = name+str(i+1)
        instanceList.append(DeepTexture( currentName, tex_path_list[i], base_img_path = base_path))
        scoreList.append(0)

def initializeProcesses(iterations_ = 2):
    for i in range(len(instanceList)):
        processList.append(Process(target=instanceList[i].runIterations, args = (iterations_,) ))

def buildTextures(features_ = 'pool'):
    for i in instanceList:
        varr = i.buildTexture(features = features_)
        print("AYYYYY2          ",varr)
        losses.append(varr)
        print(losses)

def buildTexturesWithLoss(features = 'pool'):
    for i in instanceList:
        i.buildTextureWithLoss(features)
        #print(layer_loss_scores)

def runIterations(iterations_ = 2,pInterval = 10):
    for i in range(len(instanceList)):
        scoreList[i] = instanceList[i].runIterations(iterations = iterations_,printInterval = pInterval)

def runIterationsParallel():
    for i in processList:
        i.start()
    for i in processList:
        i.join()

def calculateWeightedScore():
    # Initialization of new Score list
    newScore = []
    # Getting min and max to convert from min <= oldscore <= max to offset <= newScore <= (1-offset) so the scores contribute at least 20% and at most 80%
    min_ = min(scoreList)
    max_ = max(scoreList)
    # Initialization of offset and the subsequent variables for the conversion
    offset = 0.2
    m = 1-2*offset
    l = offset/m
    # Converting to [offset,1-offset] using  (1-2offset) * ( (x-min)/(max-min) + (offset)/(1-2*offset) ) 
    # What is done is a conversion from [min_score,max_score] to [0,1] by (x-min_score)/(max_score-min_score)
    # Then an abstract_offset is added so that 0 becomes "offset" and 1 becomes "1-offset".
    # If the initial abstract offset is called y so it becomes [y,1+y], there has to be a variable k so that y/k = offset and (1+y)/k = 1-offset
    # The solution for the system [y/k,(1+y)/k] = [offset,1-offset] is: "y = (offset)/(1-2*offset)" and "k = 1/(1-2*offset)"
    # For convienience "m = 1/k", "y = offset/m" and "new_score =   m * (score_from_0_to_1 + y)"
    for i in scoreList:
        newScore.append(m*(((i-min_)/(max_-min_))+l))
    #print("newScore before sum conversion:",newScore)

    # Getting the sum of the scores
    sum_ = sum(newScore)
    # Weighing down the scores by the sum so sum(newScore) = 1
    # This procedure is done so that when the pictures merge, there is no clipping
    for i in range(len(newScore)):
        newScore[i]/=sum_
    return newScore

# Pretty self explanatory
def printScores():
    print("Average Scores are:",scoreList)


def calculateOutput():
    # Getting the weighted scores that any pixel will be weighed, then added
    newScore = calculateWeightedScore()

    # Initialization of the created image
    images = []

    # Adding the images to the list
    for i in instanceList:
        images.append(Image.open(i.fname))

    # Creation of a modification matrix for each r g b value to multiply of each image by the weight
    for i in range(len(images)):
        matrix =   (newScore[i],    0,          0,            0,
                    0,              newScore[i],0,            0,
                    0,              0,          newScore[i],  0)
        images[i] = images[i].convert("RGB",matrix)
    
    # Getting the size and initializing the new image
    sizes = images[0].size
    new_image = Image.new('RGB',sizes,'black')

    # Getting the pixels "weird list of tuples" of the image
    pixels = new_image.load()

    # This assortment of ifs means "for each pixel on all images"
    for i in range(sizes[0]):
        for j in range(sizes[1]):
            sum_ = [0,0,0]
            for s in range(len(images)):
                # Getting image's pixel and adding it to the sum pixel
                old_pixel = images[s].getpixel((i,j))
                sum_[0] += old_pixel[0]
                sum_[1] += old_pixel[1]
                sum_[2] += old_pixel[2]
            # The final pixel value is the sum of all the weighted pixel values from the images
            pixels[i,j] = (sum_[0],sum_[1],sum_[2])

    # Creating the filename and saving the image
    file_name = name+'_at_iteration %d.png'%(instanceList[0].total_iterations)
    new_image.save(file_name)
    return




if __name__ == '__main__':
    
    # Here is a bunch of manual tests 

    #tex = DeepTexture('tex1','data/inputs/tex_ruins2.png',base_img_path="data/inputs/base_ruins.png")
    #tex.buildTexture(features='all')
    #a = tex.runIterations(iterations = 2)
    #print("for tex we have loss:",a)
    #a = tex.runIterations(iterations = 4)
    #print("for texx we have loss:",a)

    # Initialization of list of images to put through the program
    tex_list = ['data/inputs/tex_ruins2.png','data/inputs/tex_ruins1.png','data/inputs/tex_ruins3.png']
    base_img = 'data/inputs/base_ruins.png'
    # Initialization of the neural networks
    initializeList(base_img,tex_list)

    #Building full textures to determine which image to use as base
    buildTextures()
    # Building textures to determine the loss of each layer 
    buildTexturesWithLoss()
    # Getting the min index to choose photo
    #minIndex = getMinIndex()
    #print("LMAOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO: ",minIndex)
    #final_tex_img = tex_list[minIndex]

    createLoss()

    finalNetwork = DeepTexture( (name+"_final"), tex_list, base_img_path = base_img)
    instanceList.append(finalNetwork)
    
    finalNetwork.buildTexture(lossArray = finalLosses )
    
    finalNetwork.runIterations(iterations = 100)
    
    
    
    
    
    
    
    
    # Here is an attempt to use multithreading but is misused
    #initializeProcesses(4)
    #runIterationsParallel()
    
    # Runnning the iterations in series
    #runIterations(100,10)
    # Printing the scores
    #printScores()
    # Resuming by running more iterations
    #runIterations(100,10)
    # Printing final scores
    printScores()
    calculateOutput()
