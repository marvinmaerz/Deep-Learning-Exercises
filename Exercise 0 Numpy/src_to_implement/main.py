import pattern
import generator



# checker = pattern.Checker(16,4)
# checker.draw()
# checker.show()
#
#
# circle = pattern.Circle(1024, 200, (512, 256))
# circle.draw()
# circle.show()
#
#
# spectrum = pattern.Spectrum(255)
# spectrum.draw()
# spectrum.show()

label_path = 'Exercise 0 Numpy/src_to_implement/Labels.json'
file_path = 'Exercise 0 Numpy/src_to_implement/exercise_data/'
generator = generator.ImageGenerator(file_path, label_path, 10, [32, 32, 3], rotation=False, mirroring=False, shuffle=False)
# generator.next()
# generator.next()
# generator.next()
# generator.next()
generator.show()
generator.show()
generator.show()
generator.show()