class H2B:
    def __init__(self):
        self.trajectory = []
        self.time = []
        self.channel = []
        self.channel_size = 0
        self.max_radius = None
        self.file_name = None
        self.id = None
        self.predicted_label = None
        self.manuel_label = None
        self.predicted_proba = None
        self.diff_coef = None

    def get_trajectory(self):
        return self.trajectory

    def set_trajectory(self, trajectory):
        self.trajectory = trajectory

    def get_time(self):
        return self.time

    def set_time(self, time):
        self.time = time

    def get_len_trajectory(self):
        return len(self.trajectory)

    def get_time_duration(self):
        return self.time[-1] - self.time[0]

    def set_max_radius(self, radius):
        self.max_radius = radius

    def get_max_radius(self):
        return self.max_radius

    def set_channel(self, channel):
        self.channel = channel

    def get_channel(self):
        return self.channel

    def set_channel_size(self, n):
        self.channel_size = n

    def get_channel_size(self):
        return self.channel_size

    def set_id(self, id):
        self.id= id

    def get_id(self):
        return self.id

    def set_file_name(self, file_name):
        self.file_name = file_name

    def get_file_name(self):
        return self.file_name

    def set_predicted_label(self, label):
        self.predicted_label = label

    def get_predicted_label(self):
        return self.predicted_label

    def set_predicted_proba(self, proba):
        self.predicted_proba = proba

    def get_predicted_proba(self):
        return self.predicted_proba

    def set_manuel_label(self, label):
        self.manuel_label = label

    def get_manuel_label(self):
        return self.manuel_label

    def set_diff_coef(self, coef):
        self.diff_coef = coef

    def get_diff_coef(self):
        return self.diff_coef

    def copy(self):
        copy_h2b = H2B()
        copy_h2b.set_id(self.get_id())
        copy_h2b.set_trajectory(self.get_trajectory())
        copy_h2b.set_time(self.get_time())
        copy_h2b.set_channel(self.get_channel())
        copy_h2b.set_channel_size(self.get_channel_size())
        copy_h2b.set_max_radius(self.get_max_radius())
        copy_h2b.set_file_name(self.get_file_name())
        copy_h2b.set_id(self.get_id())
        copy_h2b.set_predicted_label(self.get_predicted_label())
        copy_h2b.set_manuel_label(self.get_manuel_label())
        copy_h2b.set_predicted_proba(self.get_predicted_proba())
        copy_h2b.set_diff_coef(self.diff_coef)
        return copy_h2b
