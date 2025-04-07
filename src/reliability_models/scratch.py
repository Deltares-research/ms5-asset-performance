from probabilistic_library import ReliabilityProject, DistributionType, ReliabilityMethod, StartMethod, StandardNormal


def linear(a, b):
    return 1.9 - (a+b)


if __name__ == "__main__":

    project = ReliabilityProject()
    project.model = linear

    project.variables["a"].distribution = DistributionType.uniform
    project.variables["a"].minimum = -1
    project.variables["a"].maximum = 1
