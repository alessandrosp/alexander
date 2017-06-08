import pandas as pd

class NoneTransformer():
	'''Convenience class for no transformation.'''

	def __init__(self):
		pass

	def fit(self, X, y=None):
		pass

	def transform(self, X):
		return X

	def fit_transform(self, X):
		return X


class Pipeline():

	def __init__(self, steps):
		self.steps = self._format_steps(steps)

	def _format_steps(self, steps):
		"""It formats steps.

		It guarantees that steps[1] is a list rather than a single object. Also,
		it replaces every None with a NoneTransformer.
		"""
		formatted_steps = []
		for step in steps:
			selector = step[0]
			actions = step[1]
			if not isinstance(actions, list):
				actions = [actions]
			new_actions = []
			for action in actions:
				if action is None:
					new_actions.append(NoneTransformer())
				else:
					new_actions.append(action)
			new_step = (selector, new_actions)
			formatted_steps.append(new_step)
		return formatted_steps

	def fit(self, X, y=None):
		for step in self.steps:
			selector = step[0]
			cX = pd.DataFrame(X[selector])
			for transformer in step[1][0:-1]:  # All but the last one
				cX = transformer.fit_transform(cX)
			step[1][-1].fit(cX)

	def transform(self, X):
		new_dfs = []
		for step in self.steps:
			selector = step[0]
			cX = pd.DataFrame(X[selector])
			# Here we have multiple transformers (list)
			if isinstance(step[1], list):
				for transformer in step[1]:
					cX = transformer.transform(cX)
			# Here we have only one transformer (single object)
			else:
				cX = step[1].transform(cX)
			new_dfs.append(cX)
		result = pd.concat(new_dfs, axis=1)
		return result

	def fit_transform(self, X):
		self.fit(X)  # TODO(): This is inefficient
		result = self.transform(X)
		return result
