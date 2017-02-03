class Transform(object):

    def transform(self, x, y, Z):
        raise NotImplementedError('Transform.transform')

    def reverse(self, *args, **kwargs):
        raise NotImplementedError('Transform.reverse')