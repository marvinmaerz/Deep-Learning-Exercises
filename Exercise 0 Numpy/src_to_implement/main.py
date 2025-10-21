import pattern
import generator



checker = pattern.Checker(16,4)
checker.draw()
checker.show()


circle = pattern.Circle(1024, 200, (512, 256))
circle.draw()
circle.show()

# generator = generator.Generator()