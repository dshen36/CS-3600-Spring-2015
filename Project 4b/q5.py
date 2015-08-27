import Testing
import NeuralNetUtil
import NeuralNet

print "----------------- testPenData --------------------"
i = 0
acclist = []
while i < 5:
	print "running iteration #", i+1
	nnet, testAccuracy = Testing.testPenData()
	acclist.append(testAccuracy)
	i = i + 1

print "Iterations finished"
print "accuracy average:", Testing.average(acclist)
print "accuracy standard deviation:", Testing.stDeviation(acclist)
print "max accuracy:", max(acclist)

print "----------------- testCarData --------------------"
i = 0
acclist = []
while i < 5:
	print "running iteration #", i+1
	nnet, testAccuracy = Testing.testCarData()
	acclist.append(testAccuracy)
	i = i + 1

print "Iteration finished"
print "accuracy average:", Testing.average(acclist)
print "accuracy standard deviation:", Testing.stDeviation(acclist)
print "max accuracy:", max(acclist)
