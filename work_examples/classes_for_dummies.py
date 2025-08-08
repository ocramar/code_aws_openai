class Person:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hi, I'm {self.name}"
    
class Student(Person):
    def study(self):
        return f"{self.name} is studying."
    
p = Person("Alice")
print(p.greet())        # OK
# print(p.study())      ❌ ERROR — Person doesn't know how to study

s = Student("Bob")
print(s.greet())        # ✅ Inherited from Person
print(s.study())        # ✅ Defined in Student