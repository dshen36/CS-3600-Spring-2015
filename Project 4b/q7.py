import Testing
import NeuralNetUtil
import NeuralNet

numNeurons = 0
while 1:
	print "--------------running with", numNeurons , "neurons per hidden layer------------------"
	i = 0
	acclist = []
	while i < 5:
		print "running iteration #", i+1
		nnet, testAccuracy = NeuralNet.buildNeuralNet(Testing.xorData,maxItr = 200, hiddenLayerList = [numNeurons])
		acclist.append(testAccuracy)
		i = i + 1

	print "Iterations finished"
	print "accuracy average:", Testing.average(acclist)
	print "accuracy standard deviation:", Testing.stDeviation(acclist)
	print "max accuracy:", max(acclist)
	if Testing.average(acclist) == 1:
		break
	numNeurons = numNeurons + 1