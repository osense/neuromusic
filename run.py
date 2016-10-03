import net
import datahandler

n = net.Net()
n.load()

for i in range(1, 46):
	a = datahandler.get('piano', i)
	n.learn(a)

n.save()