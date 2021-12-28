import math
import fractions
print('')
a = float(input("Put in the value for a: "))
b = float(input("Put in the value for b: "))
c = float(input("Put in the value for c: "))

pos_root = ((-b+(b**2-4*a*c)**(1/2))/(2*a))
neg_root = ((-b-(b**2-4*a*c)**(1/2))/(2*a))

print('')
print("Positive root: ", (pos_root))
print("Negative root: ", (neg_root))
print('')