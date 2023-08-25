import timeit
import statistics as stats

times = timeit.repeat('rust_methods.get_sphere_radii(nc, vc, indices)', setup='import rust_methods, numpy; nc = numpy.loadtxt("./output/cnts.csv", delimiter=", "); vc = numpy.loadtxt("./output/N.csv", delimiter=", "); indices = list(range(0,nc.shape[0]));', number=100, repeat=100)
mean = stats.fmean(times) / 100
stdev = stats.stdev(times) / 100
print("On average it took {:.2e} seconds with a standard deviation of {:.2e} seconds.".format(mean, stdev))