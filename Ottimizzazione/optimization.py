 # -*- coding: utf-8 -*-
import time
import random
import math
import struct

people = [('Seymour','BOS'),
          ('Franny','DAL'),
          ('Zooey','CAK'),
          ('Walt','MIA'),
          ('Buddy','ORD'),
          ('Les','OMA')]
# Laguardia
destination='LGA'
maxvalue=100
minvalues=-100
flights={}

#for line in file('schedule.txt'):
  #origin,dest,depart,arrive,price=line.strip().split(',')
  #flights.setdefault((origin,dest),[])

  # Add details to the list of possible flights
  #flights[(origin,dest)].append((depart,arrive,int(price)))

def getminutes(t):
  x=time.strptime(t,'%H:%M')
  return x[3]*60+x[4]

def printschedule(r): #stampa schedule dei voli di andata e ritorno
  for d in range(len(r)/2):
    name=people[d][0]
    origin=people[d][1]
    out=flights[(origin,destination)][int(r[d*2])]
    ret=flights[(destination,origin)][int(r[d*2+1])]
    print '%10s%10s %5s-%5s $%3s %5s-%5s $%3s' % (name,origin,
                                                  out[0],out[1],out[2],
                                                  ret[0],ret[1],ret[2])

def schedulecost(sol):
  totalprice=0
  latestarrival=0
  earliestdep=24*60

  for d in range(len(sol)/2):
    # Get the inbound and outbound flights
    origin=people[d][1]
    outbound=flights[(origin,destination)][int(sol[d*2])]
    returnf=flights[(destination,origin)][int(sol[d*2+1])]
    
    # Total price is the price of all outbound and return flights
    totalprice+=outbound[2]
    totalprice+=returnf[2]
    
    # Track the latest arrival and earliest departure
    if latestarrival<getminutes(outbound[1]): latestarrival=getminutes(outbound[1])
    if earliestdep>getminutes(returnf[0]): earliestdep=getminutes(returnf[0])
  
  # Every person must wait at the airport until the latest person arrives.
  # They also must arrive at the same time and wait for their flights.
  totalwait=0  
  for d in range(len(sol)/2):
    origin=people[d][1]
    outbound=flights[(origin,destination)][int(sol[d*2])]
    returnf=flights[(destination,origin)][int(sol[d*2+1])]
    totalwait+=latestarrival-getminutes(outbound[1])
    totalwait+=getminutes(returnf[0])-earliestdep  

  # Does this solution require an extra day of car rental? That'll be $50!
  if latestarrival>earliestdep: totalprice+=50
  
  return totalprice+totalwait


def randomoptimize(domain,costf):
  best=999999999
  bestr=None
  for i in range(0,1000):
    # Create a random solution
    r=[float(random.randint(domain[i][0],domain[i][1])) 
       for i in range(len(domain))]
    
    # Get the cost
    cost=costf(r)
    
    # Compare it to the best one so far
    if cost<best:
      best=cost
      bestr=r 
  return r

def hillclimb(domain,costf):
  # Create a random solution
  sol=[random.randint(domain[i],domain[i])
      for i in range(len(domain))]
  # Main loop
  while 1:
    # Create list of neighboring solutions
    neighbors=[]
    
    for j in range(len(domain)):
      # One away in each direction
      if sol[j]>domain[j]:
        neighbors.append(sol[0:j]+[sol[j]+1]+sol[j+1:])
      if sol[j]<domain[j]:
        neighbors.append(sol[0:j]+[sol[j]-1]+sol[j+1:])
    # print neighbors
    # print '\n'
    # print sol
    # See what the best solution amongst the neighbors is
    current=costf
    best=current
    for j in range(len(neighbors)):
      cost=costf(neighbors[j])
      if cost<best:
        best=cost
        sol=neighbors[j]

    # If there's no improvement, then we've reached the top
    if best==current:
      break

  fig, ax = plt.subplots()
  line,=ax.plot(domain, best,'r-', linestyle="solid", marker="o")
  plt.xlim(-100, 120)
  plt.ylim(-1000, 17000)
  plt.xlabel('valore di x')
  plt.ylabel('valore di F(x)')
  line.set_antialiased(False)
  plt.title("Grafico Hill Climbing")
  plt.grid()
  plt.show()

  return sol

def annealingoptimize(domain,costf,T=10000.0,cool=0.95,step=1):
  # Initialize the values randomly
  vec=[float(random.randint(domain[i][0],domain[i][1])) 
       for i in range(len(domain))]
  
  while T>0.1:
    # Choose one of the indices
    i=random.randint(0,len(domain)-1)

    # Choose a direction to change it
    dir=random.randint(-step,step)

    # Create a new list with one of the values changed
    vecb=vec[:]
    vecb[i]+=dir
    if vecb[i]<domain[i][0]: vecb[i]=domain[i][0]
    elif vecb[i]>domain[i][1]: vecb[i]=domain[i][1]

    # Calculate the current cost and the new cost
    ea=costf(vec)
    eb=costf(vecb)
    p=pow(math.e,(-eb-ea)/T)

    # Is it better, or does it make the probability
    # cutoff?
    if (eb<ea or random.random()<p):
      vec=vecb      

    # Decrease the temperature
    T=T*cool
  return vec

def binary(num):
    # Struct can provide us with the float packed into bytes. The '!' ensures that
    # it's in network byte order (big-endian) and the 'f' says that it should be
    # packed as a float. Alternatively, for double-precision, you could use 'd'.
    packed = struct.pack('!f', num[0])
    #print 'Packed: %s' % repr(packed)

    # For each character in the returned string, we'll turn it into its corresponding
    # integer code point
    # 
    # [62, 163, 215, 10] = [ord(c) for c in '>\xa3\xd7\n']
    integers = [ord(c) for c in packed]
    #print 'Integers: %s' % integers

    # For each integer, we'll convert it to its binary representation.
    binaries = [bin(i) for i in integers]
    #print 'Binaries: %s' % binaries

    # Now strip off the '0b' from each of these
    stripped_binaries = [s.replace('0b', '') for s in binaries]
    #print 'Stripped: %s' % stripped_binaries

    # Pad each byte's binary representation's with 0's to make sure it has all 8 bits:
    #
    # ['00111110', '10100011', '11010111', '00001010']
    padded = [s.rjust(8, '0') for s in stripped_binaries]
    #print 'Padded: %s' % padded

    # At this point, we have each of the bytes for the network byte ordered float
    # in an array as binary strings. Now we just concatenate them to get the total
    # representation of the float:
    return ''.join(padded)

def bits2int(bits):
    # You may want to change ::-1 if depending on which bit is assumed
    # to be most significant. 
    bits = [int(x) for x in bits[::-1]]

    x = 0
    for i in range(len(bits)):
        x += bits[i]*2**i
    return x
  
def as_float32(self):
    f = int(self, 2)
    return struct.unpack('f', struct.pack('I', f))[0]

#Procedura che definisce la funzione da ottimizzare 
# con gli algoritmi di ricerca locale  
import math
def F(x):
    if(x<=5.2):
        return 10
    if((x>=5.2)and(x<=20)):
        return x*x
    if(x>20):
        return 160 * x + math.cos(x*180/math.pi)

#Procedura che calcola il costo dell'algoritmo
# di ricerca per trovare il massimo locale
def cost_fun(sol):
  cost = minvalues
  for i in range(0, len(sol)):
    current=maxvalue-F(sol[i]);
    if current>cost:
      if current < 0:
        current=current*(-1)+100
      cost = current
  return cost

#Procedura che definisce la funzione di costo 
# che permette all'algoritmo di trovare un massimo locale
def cost_max(sol):
  cost=[]
  for i in range(0, len(sol)):
    if sol[i]>100 or sol[i]<-100:
        cost.append(0)
    else:
        cost.append(F(sol[i]))
  return cost

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
def geneticoptimize(domain,costf,popsize=100,step=1,
                    mutprob=0.2,elite=0.2,maxiter=100):
  # Mutation Operation
  def mutate(vec):
    i=random.randint(0,len(vec)-1)
    if random.random()<0.5:
      if vec[i]=='0':
        #vec=vec[0:i]+'1'+vec[i+1:] 
        #if(round(as_float32(vec),2)<-100 or round(as_float32(vec),2)>100)
        return vec[0:i]+'1'+vec[i+1:] 
      else:
        return vec[0:i]+'0'+vec[i+1:]
  
  # Crossover Operation
  def crossover(r1,r2):
    i=random.randint(1,len(r1))
    return r1[0:i]+r2[i:]

  # Build the initial population
  x=[]
  sc=[]
  pop=[]
  for i in range(popsize):
    vec=[round(random.uniform(domain[i],domain[i]),2) 
         for i in range(len(domain))]
    pop.append(vec)
  
  # How many winners from each generation?
  topelite=int(elite*popsize)
  # Main loop 
  for i in range(maxiter):
    scores=[(costf,v) for v in pop if v is not None ]
    scores.sort(reverse=True)
    ranked=[v for (s,v) in scores]
    for i in range(0,len(ranked)):
      ranked[i]=binary(ranked[i]);
    #for i in range(0,len(vec)):
    #  vecbin[i]=binary([vec[i]])
    #print vec
    #for i in range(0,len(vec)):
    #  vec[i]=round(as_float32(vec[i]),2)
    #print vec
    # Start with the pure winners
    pop=ranked[0:topelite]
    # Add mutated and bred forms of the winners
    while len(pop)<popsize:
      if random.random()<mutprob:

        # Mutation
        c=random.randint(0,topelite)
        pop.append(mutate(ranked[c]))
      else:
      
        # Crossover
        c1=random.randint(0,topelite)
        c2=random.randint(0,topelite)
        pop.append(crossover(ranked[c1],ranked[c2]))
    
    # Print current best score
    # print "score"
    # print scores[0][0]
    x=x+[scores[0][0]]
    sc+=[scores[0][1]]
    #print "pop"
    for i in range(0,len(pop)):
      if pop[i] is not None:
        pop[i]=[round(as_float32(pop[i]),2)]
        if math.isnan(float(pop[i][0]))==True:
          pop[i]=[0.00]
    #print "pop"
  fig, ax = plt.subplots()
  line,=ax.plot(domain, scores[0][0],'r-', linestyle="solid", marker="o")
  plt.xlim(-100, 120)
  plt.ylim(-1000, 17000)
  plt.xlabel('valore di x')
  plt.ylabel('valore di F(x)')
  line.set_antialiased(False)
  plt.title("Grafico Genetic Algorithm")
  plt.grid()
  plt.show()
  return scores[0][1]

if __name__ == '__main__':

    #Popolazione iniziale dei valori da assegnare alla funzione
    popolazione = [-60, 0, 5, 10, 15, 80, 100, 110]

    print('\n\nPopolazione iniziale: ' + str(popolazione))

    #Valori di ottimizzazione della funzione
    f = cost_max(popolazione)

    print('Optimization: ' + str(f))

    #Calcoliamo il costo dell'algoritmo data la popolazione
    costo_max = cost_fun(popolazione)

    print('Costo dello algoritmo: ' + str(costo_max) + '\n\n')

    #Funzione del Capitolo 5che implementa l'algoritmo genetico
    geneticoptimize(popolazione, f)

    #Funzione del Capitolo 5 che implementa l'algoritmo Hill Climbing
    hillclimb(popolazione, f)