from Libs.classifier import ParallelProcess

if __name__ == '__main__':
	print('Running baseline')
	ParallelProcess().run()
	print('Running similarity')
	ParallelProcess().run(True)
