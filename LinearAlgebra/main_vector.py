from LinearAlgebra.playLA.Vector import Vector

if __name__ == "__main__":

    vec = Vector([5, 2])
    print(vec)
    print(len(vec))
    print("vec[0] = {}, vec[1] = {}".format(vec[0], vec[1]))

    vec2 = Vector([3, 1])
    print("{} + {} = {}".format(vec, vec2, vec + vec2))

    vec3 = Vector([4, 1])
    print("{} - {} = {}".format(vec, vec3, vec - vec3))

    vec4 = Vector([5, 1])
    print("{} * {} = {}".format(3, vec4, 3 * vec4))

    vec5 = Vector([5, 1])
    print("{} * {} = {}".format(vec4, 4, vec4 * 4))

