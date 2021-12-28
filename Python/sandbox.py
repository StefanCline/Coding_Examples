class Dog():
    
    def __init__(self,breed):
        self.breed = breed

my_dog = Dog(breed = 'Lab')

print(type(my_dog))
print(my_dog.breed)
