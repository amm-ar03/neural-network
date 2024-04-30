import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#this function is just the derivative of sigmoid above
def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1-fx)

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

class Neural_Network:

    def __init__(self):
        #Weights 
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * x[0] + self.w6 * x[1] + self.b3)
        return o1
    
    def train(self, data, all_y_trues):

        learn_rate = 0.1
        epochs = 1000

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * x[0] + self.w6 * x[1] + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1
                #partial derivative
                L_ypred = -2 * (y_true - y_pred)

                #Neuron o1
                ypred_w5 = h1 * deriv_sigmoid(sum_o1)
                ypred_w6 = h2 * deriv_sigmoid(sum_o1)
                ypred_b3 = deriv_sigmoid(sum_o1)

                y_pred_h1 = self.w5 * deriv_sigmoid(sum_o1)
                y_pred_h2 = self.w6 * deriv_sigmoid(sum_o1)

                #Neuron h1
                h1_w1 = x[0] * deriv_sigmoid(sum_h1)
                h1_w2 = x[1] * deriv_sigmoid(sum_h1)
                h1_b1 = deriv_sigmoid(sum_h1)

                #Neuron h2
                h2_w3 = x[0] * deriv_sigmoid(sum_h2)
                h2_w4 = x[1] * deriv_sigmoid(sum_h2)
                h2_b2 = deriv_sigmoid(sum_h2)

                #Updating weights and biases
                #Neuron h1
                self.w1 -= learn_rate * L_ypred * y_pred_h1 * h1_w1
                self.w2 -= learn_rate * L_ypred * y_pred_h1 * h1_w2
                self.b1 -= learn_rate * L_ypred * y_pred_h1 * h1_b1

                #Neuron h2
                self.w3 -= learn_rate * L_ypred * y_pred_h2 * h2_w3
                self.w4 -= learn_rate * L_ypred * y_pred_h2 * h2_w4
                self.b2 -= learn_rate * L_ypred * y_pred_h2 * h2_b2

                #Neuron o1
                self.w5 -= learn_rate * L_ypred * ypred_w5
                self.w6 -= learn_rate * L_ypred * ypred_w6
                self.b3 -= learn_rate * L_ypred * ypred_b3

                #This calculates total loss at end of each epoch 
            if epoch % 10 == 0:
                y_pred = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_pred)
                print("Epoch %d loss: %.3f" % (epoch, loss))

data = np.array([
    [-7, -3], #person 1
    [40, 4], #person 2
    [-7, 3], #person 3 
    [120, 9], #person 4
])
all_y_trues = np.array([
    1,
    0,
    0,
    0,
])

network = Neural_Network()
network.train(data, all_y_trues)

#shift amounts chosen is 135 pounds and 66 inches 

mike_wazowski = np.array([6, -6]) #141 pounds, 60 inches
phineas = np.array([19, 3]) #154 pounds, 69 inches

print("Mike Wazowski: %.3f " % network.feedforward(mike_wazowski))
print("Phineas: %.3f " % network.feedforward(phineas))
                
