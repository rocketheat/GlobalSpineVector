import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import tkinter.simpledialog as simpledialog

from PIL import Image, ImageTk

import scipy
from scipy.interpolate import CubicSpline

import numpy as np
import pandas as pd

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Global Spine Vector Application V2.0")

        self.canvas = tk.Canvas(root)
        self.canvas.pack(fill=tk.BOTH, expand=tk.YES)
        self.canvas.bind("<Button-1>", self.get_coords)


        menu = tk.Menu(root)
        root.config(menu=menu)

        file_menu = tk.Menu(menu)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New", command=self.restart_app)
        file_menu.add_command(label="Open...", command=self.open_image)
        file_menu.add_command(label="Exit", command=root.quit)
        file_menu.add_command(label="Save Data", command=self.save_table)

        # Reset Functionality
        self.points = []
        self.coordinates = {}
        self.level = []
        self.tags = []
        self.point_dict = {}
        self.text_dict = {}
        self.texts = []

        edit_menu = tk.Menu(menu)
        menu.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Resize Image", command=self.resize_image)
        edit_menu.add_command(label="Reset Points", command=self.reset_points)
        edit_menu.add_command(label="Draw Spline", command=self.draw_spline)
        edit_menu.add_command(label="Spine Vector", command=self.spine_vector)
        edit_menu.add_command(label="Show Table", command=self.show_table)
        edit_menu.add_command(label="Delete Point", command=self.delete_point)

        # Label
        self.label = tk.Label(root, text="Enter Patient's Weight in kg:")
        self.label.pack(pady=10)

        # Entry widget
        self.weight = tk.StringVar()
        self.entry_widget = tk.Entry(root, textvariable=self.weight)
        self.entry_widget.pack(pady=10)

        # Label
        self.label_P = tk.Label(root, text="Enter Point to Delete:")
        self.label_P.pack(pady=10)

        # Entry widget
        self.delete_p = tk.StringVar()
        self.entry_widget = tk.Entry(root, textvariable=self.delete_p)
        self.entry_widget.pack(pady=10)

        self.level_proportion = {
            'C1': 1.,
            'C2': 1.,
            'C3': 1.,
            'C4': 1.,
            'C5': 1.,
            'C6': 1.,
            'C7': 1.,
            'T1': 1.1,
            'T2': 1.1,
            'T3': 1.4,
            'T4': 1.3,
            'T5': 1.3,
            'T6': 1.3,
            'T7': 1.4,
            'T8': 1.5,
            'T9': 1.6,
            'T10': 2.0,
            'T11': 2.1,
            'T12': 2.5,
            'L1': 2.4,
            'L2': 2.4,
            'L3': 2.3,
            'L4': 2.6,
            'L5': 2.6,
            'S1': 2.6
        }
        self.calc_cum_level()



    def calc_cum_level(self):
        self.cumulative_level_proportion = {}
        self.cumulative_single_level_proportion = {}

        total_head_contribution = 7
        total_cervical_contribution = 4
        total_weight_on_lumbar = 65
        total_trunk_upper_extremities_contribution = total_weight_on_lumbar - total_head_contribution - total_cervical_contribution

        cervical_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
        thoracic_lumbar_list = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6' ,'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5', 'S1']

        sum_cervical_regional_contribution = 0
        for l in cervical_list:
            sum_cervical_regional_contribution += self.level_proportion[l]
        for l in cervical_list:
            self.cumulative_level_proportion[l] = self.level_proportion[l] / sum_cervical_regional_contribution
            self.cumulative_single_level_proportion[l] = self.level_proportion[l] / sum_cervical_regional_contribution
        prior_c_contrib = total_head_contribution
        for i, l in enumerate(cervical_list):
            self.cumulative_level_proportion[l] = prior_c_contrib + (total_cervical_contribution * self.cumulative_level_proportion[l])
            self.cumulative_single_level_proportion[l] = (total_cervical_contribution * self.cumulative_single_level_proportion[l])

        sum_thoracic_lumbar_regional_contribution = 0
        for l in thoracic_lumbar_list:
            sum_thoracic_lumbar_regional_contribution += self.level_proportion[l]
        for l in thoracic_lumbar_list:
            self.cumulative_level_proportion[l] = self.level_proportion[l] / sum_thoracic_lumbar_regional_contribution
            self.cumulative_single_level_proportion[l] = self.level_proportion[l] / sum_thoracic_lumbar_regional_contribution
        prior_tl_contrib = total_head_contribution + total_cervical_contribution
        for i, l in enumerate(thoracic_lumbar_list):
            self.cumulative_level_proportion[l] = prior_tl_contrib + (total_trunk_upper_extremities_contribution * self.cumulative_level_proportion[l])
            self.cumulative_single_level_proportion[l] = (total_trunk_upper_extremities_contribution * self.cumulative_single_level_proportion[l])
            prior_tl_contrib = self.cumulative_level_proportion[l]


    def open_image(self):
        file_path = filedialog.askopenfilename()
        self.file_path = file_path.rpartition('/')[0]

        if not file_path:
            return

        self.image = Image.open(file_path)

        self.tk_image = ImageTk.PhotoImage(self.image)

        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def resize_image(self):
        # Calculate the scale factor needed to fit the image into the window, while preserving its aspect ratio
        scale_factor = min(self.root.winfo_width() / self.tk_image.width(), self.root.winfo_height() / self.tk_image.height())

        # Calculate the new dimensions of the resized image
        width = int(self.tk_image.width() * scale_factor * 0.75)
        height = int(self.tk_image.height() * scale_factor * 0.75)

        # Resize the image and update the label
        scaled_image = self.image.resize((width, height))
        self.tk_image = ImageTk.PhotoImage(scaled_image)

        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def get_coords(self, event):
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        # Prompt user for point tag text name
        tag_name = simpledialog.askstring("Input", "Enter the point tag text name:", parent=self.root)
        if tag_name:  # If user provided a name and didn't cancel the dialog
            text_id = self.canvas.create_text(x - 25, y, text=tag_name, anchor=tk.W, tags=tag_name)
            self.texts.append(text_id)
            self.text_dict[tag_name] = text_id
            self.level.append(tag_name)

        # Draw a small circle at the clicked location
        radius = 5  # You can adjust this value as needed
        point_id = self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="red")
        self.point_dict[tag_name] = point_id

        self.points.append(point_id)
        self.coordinates[tag_name] = (x, y)

        # If you want to save to a file, you can append the coordinates to a file
        # with open('coordinates.txt', 'a') as f:
        #     f.write(f"x={x}, y={y}\n")

    def _reset_points(self):
        self.canvas.delete("spline")  # Deleting spline if it exists
        self.canvas.delete("angle_text")
        self.canvas.delete("point_tag")
        self.canvas.delete("spine_vec")
        self.canvas.delete("vector_text")

    def reset_points(self):
        for point_id in self.points:
            self.canvas.delete(point_id)
        for text_id in self.texts:
            self.canvas.delete(text_id)
        self.points = []
        self.coordinates = {}
        self.canvas.delete("spline")  # Deleting spline if it exists
        self.canvas.delete("angle_text")
        self.canvas.delete("point_tag")
        self.canvas.delete("spine_vec")
        self.canvas.delete("vector_text")

    def save_table(self):
        df = pd.DataFrame(self.stored_data)
        df.columns = ['Slope Angle', 'Shear Vector Magnitude', 'Normal Vector Magnitude', 'Vector Ratio', 'Level']
        df.to_csv(self.file_path + '/spine_vec.csv', index=False)
        df = pd.DataFrame(self.coordinates)
        df = df.T.reset_index().rename(columns={"index": "Level", 0: "X", 1: "Y"})
        print(df)
        df.to_csv(self.file_path + '/coordinates.csv', index=False)

    def draw_spline(self):
        if len(self.coordinates) < 2:
            print("Need at least two points to draw a spline.")
            return

        self._reset_points()

        temp_coordinates = list(self.coordinates.values())
        temp_coordinates.sort(key=lambda x: x[1])# Sorting by y values
        x_coords, y_coords = zip(*temp_coordinates)

        x = np.array(x_coords)
        y = np.array(y_coords)

        cs = CubicSpline(y, x, bc_type='natural')

        ynew = np.linspace(min(y_coords), max(y_coords), 1000)
        xnew = cs(ynew)

        for i in range(1, len(xnew)):
            self.canvas.create_line(xnew[i - 1], ynew[i - 1], xnew[i], ynew[i], fill="blue", tags="spline")

        # Calculate angles for the clicked points
        self.angles = self.calculate_angles(cs, y)

        # Display angles near the corresponding points
        for (xi, yi, angle) in zip(x, y, self.angles):
            self.canvas.create_text(xi + 15, yi, text=f"{angle:.2f}°", anchor=tk.W, tags='angle_text')

    def calculate_angles(self, spline, x_values):
        dx = spline.derivative(1)(x_values)  # First derivative
        dy = np.ones_like(dx)
        tangent_vectors = np.stack((dx, dy), axis=-1)
        normalized_tangent_vectors = tangent_vectors / np.linalg.norm(tangent_vectors, axis=-1, keepdims=True)
        angles = np.arctan2(normalized_tangent_vectors[:, 0], normalized_tangent_vectors[:, 1])
        return np.degrees(angles)

    def spine_vector(self):
        if len(self.coordinates) < 2:
            print("Need at least two points to draw a spline.")
            return

        self._reset_points()

        self.level = [item for _, item in sorted(zip([tup[1] for tup in list(self.coordinates.values())], self.level))]
        temp_coordinates = list(self.coordinates.values())
        temp_coordinates.sort(key=lambda x: x[1])# Sorting by y values
        x_coords, y_coords = zip(*temp_coordinates)

        x = np.array(x_coords)
        y = np.array(y_coords)

        cs = CubicSpline(y, x, bc_type='natural')

        ynew = np.linspace(min(y_coords), max(y_coords), 1000)
        xnew = cs(ynew)

        # Calculate angles for the clicked points
        self.angles = self.calculate_angles(cs, y)

        def calculate_vector(weight, level, angle):
            return np.abs(scipy.constants.g * weight * np.sin(np.radians(angle)) * level/58)
        def calculate_vector_normal(weight, level, angle):
            return np.abs(scipy.constants.g * weight * np.cos(np.radians(angle)) * level/58)

        def calculate_vector_S_non_abs(weight, level, angle):
            return scipy.constants.g * weight * np.sin(np.radians(angle)) * level/58
        def calculate_vector_O_non_abs(weight, level, angle):
            return scipy.constants.g * weight * np.cos(np.radians(angle)) * level/58

        w = float(self.weight.get()) if self.weight is not None else 60

        vec_mag = [calculate_vector(w, self.cumulative_single_level_proportion[l], a) for a, l in zip(self.angles, self.level)]
        vec_mag_normal = [calculate_vector_normal(w, self.cumulative_single_level_proportion[l], a) for a, l in
                   zip(self.angles, self.level)]

        vec_mag_S_non_abs = [calculate_vector_S_non_abs(w, self.cumulative_level_proportion[l], a) for a, l in zip(self.angles, self.level)]
        vec_mag_O_non_abs = [calculate_vector_O_non_abs(w, self.cumulative_level_proportion[l], a) for a, l in zip(self.angles, self.level)]

        x_e = [m * np.cos(np.radians(ang)) if np.radians(ang) < 0 else -1 * m * np.cos(np.radians(ang)) for x_0, m, ang in zip(x, vec_mag, self.angles)]
        y_e = [-1 * m * np.sin(np.radians(ang)) if np.radians(ang) < 0 else m * np.sin(np.radians(ang)) for y_0, m, ang in zip(y, vec_mag, self.angles)]

        x_e_n = [m * np.sin(np.radians(ang)) if np.radians(ang) < 0 else m * np.sin(np.radians(ang)) for x_0, m, ang in zip(x, vec_mag_normal, self.angles)]
        y_e_n = [m * np.cos(np.radians(ang)) if np.radians(ang) < 0 else m * np.cos(np.radians(ang)) for y_0, m, ang in zip(y, vec_mag_normal, self.angles)]

        vectors_with_starting_coordinates = [
            {'start': (x_0, y_0), 'vector': (x_t, y_t), 'label': l} for x_0, y_0, x_t, y_t, l in zip(x, y, x_e, y_e, self.level)
        ]
        vectors_with_starting_coordinates_Normal = [
            {'start': (x_0, y_0), 'vector': (x_t, y_t), 'label': l} for x_0, y_0, x_t, y_t, l in
            zip(x, y, x_e_n, y_e_n, self.level)
        ]

        # Regional Vectors
        cervical_levels = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
        thoracic_levels = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12']
        lumbar_levels = ['L1', 'L2', 'L3', 'L4', 'L5', 'S1']

        vectors_with_starting_coordinates_cervical = [
            {'start': (x_0, y_0), 'vector': (x_t, y_t), 'label': l} for x_0, y_0, x_t, y_t, l in
            zip(x, y, x_e, y_e, self.level) if l in cervical_levels
        ]
        vectors_with_starting_coordinates_Normal_cervical = [
            {'start': (x_0, y_0), 'vector': (x_t, y_t), 'label': l} for x_0, y_0, x_t, y_t, l in
            zip(x, y, x_e_n, y_e_n, self.level) if l in cervical_levels
        ]

        vectors_with_starting_coordinates_thoracic = [
            {'start': (x_0, y_0), 'vector': (x_t, y_t), 'label': l} for x_0, y_0, x_t, y_t, l in
            zip(x, y, x_e, y_e, self.level) if l in thoracic_levels
        ]
        vectors_with_starting_coordinates_Normal_thoracic = [
            {'start': (x_0, y_0), 'vector': (x_t, y_t), 'label': l} for x_0, y_0, x_t, y_t, l in
            zip(x, y, x_e_n, y_e_n, self.level) if l in thoracic_levels
        ]

        vectors_with_starting_coordinates_lumbar = [
            {'start': (x_0, y_0), 'vector': (x_t, y_t), 'label': l} for x_0, y_0, x_t, y_t, l in
            zip(x, y, x_e, y_e, self.level) if l in lumbar_levels
        ]
        vectors_with_starting_coordinates_Normal_lumbar = [
            {'start': (x_0, y_0), 'vector': (x_t, y_t), 'label': l} for x_0, y_0, x_t, y_t, l in
            zip(x, y, x_e_n, y_e_n, self.level) if l in lumbar_levels
        ]

        resultant_vector_cervical = np.sum([np.array(vec['vector']) for vec in vectors_with_starting_coordinates_cervical], axis=0)
        resultant_vector_normal_cervical = np.sum([np.array(vec['vector']) for vec in vectors_with_starting_coordinates_Normal_cervical],
                                         axis=0)
        resultant_vector_thoracic = np.sum([np.array(vec['vector']) for vec in vectors_with_starting_coordinates_thoracic], axis=0)
        resultant_vector_normal_thoracic = np.sum([np.array(vec['vector']) for vec in vectors_with_starting_coordinates_Normal_thoracic],
                                         axis=0)
        resultant_vector_lumbar = np.sum([np.array(vec['vector']) for vec in vectors_with_starting_coordinates_lumbar], axis=0)
        resultant_vector_normal_lumbar = np.sum([np.array(vec['vector']) for vec in vectors_with_starting_coordinates_Normal_lumbar],
                                         axis=0)

        sum_mag_cervical = round(np.linalg.norm(resultant_vector_cervical), 0)
        sum_mag_normal_cervical = round(np.linalg.norm(resultant_vector_normal_cervical), 0)

        sum_mag_thoracic = round(np.linalg.norm(resultant_vector_thoracic), 0)
        sum_mag_normal_thoracic = round(np.linalg.norm(resultant_vector_normal_thoracic), 0)

        sum_mag_lumbar = round(np.linalg.norm(resultant_vector_lumbar), 0)
        sum_mag_normal_lumbar = round(np.linalg.norm(resultant_vector_normal_lumbar), 0)

        # Calculate the angle in radians
        angle_radians_cervical = np.arctan2(resultant_vector_cervical[1], resultant_vector_cervical[0])
        # Convert to degrees
        angle_degrees_cervical = round(np.degrees(angle_radians_cervical), 0)

        # Calculate the angle in radians
        angle_radians_thoracic = np.arctan2(resultant_vector_thoracic[1], resultant_vector_thoracic[0])
        # Convert to degrees
        angle_degrees_thoracic = round(np.degrees(angle_radians_thoracic), 0)

        # Calculate the angle in radians
        angle_radians_lumbar = np.arctan2(resultant_vector_lumbar[1], resultant_vector_lumbar[0])
        # Convert to degrees
        angle_degrees_lumbar = round(np.degrees(angle_radians_lumbar), 0)
        #=======================================

        std_width = self.canvas.winfo_width() - 200
        std_height = 100

        # Extracting only the vector components and adding them

        resultant_vector = np.sum([np.array(vec['vector']) for vec in vectors_with_starting_coordinates], axis=0)
        resultant_vector_normal = np.sum([np.array(vec['vector']) for vec in vectors_with_starting_coordinates_Normal], axis=0)

        # Calculate the angle in radians
        angle_radians = np.arctan2(resultant_vector[1], resultant_vector[0])

        # Convert to degrees
        angle_degrees = round(np.degrees(angle_radians), 0)

        # Calculating magnitude
        sum_mag = round(np.linalg.norm(resultant_vector), 0)
        sum_mag_normal = round(np.linalg.norm(resultant_vector_normal), 0)

        vec_ratio = np.tan(np.radians(self.angles))

        self.stored_data = [[round(float(ang),1), round(float(mag_s),1), round(float(mag_O),1), round(float(ratio), 1), level] for ang, mag_s, mag_O, ratio, level in zip(self.angles, vec_mag_S_non_abs, vec_mag_O_non_abs, vec_ratio, self.level)]
        self.stored_data.append([180 - angle_degrees_cervical, sum_mag_cervical, sum_mag_normal_cervical, round(np.tan(angle_radians_cervical),1), 'RSV-C'])
        self.stored_data.append([180 - angle_degrees_thoracic, sum_mag_thoracic, sum_mag_normal_thoracic, round(np.tan(angle_radians_thoracic), 1), 'RSV-T'])
        self.stored_data.append([180 - angle_degrees_lumbar, sum_mag_lumbar, sum_mag_normal_lumbar, round(np.tan(angle_radians_lumbar), 1), 'RSV-L'])
        self.stored_data.append([180 - angle_degrees, sum_mag, sum_mag_normal, round(np.tan(angle_radians), 1), 'GSV'])

        self.canvas.create_line(std_width, std_height, resultant_vector[0] + std_width, resultant_vector[1] + std_height, arrow=tk.LAST, fill="blue", tags="spine_vec")
        self.canvas.create_text(resultant_vector[0] + std_width + 15, resultant_vector[1] + std_height + 15, text=f"The vector angle: {180-angle_degrees:.2f}°, \nThe vector magnitude: {sum_mag:.2f} Newton", anchor=tk.W, tags='vector_text')

        return angle_degrees, sum_mag

    def show_table(self):
        # Create and pack the treeview widget if not already existing
        if not hasattr(self, "tree"):
            self.tree = ttk.Treeview(self.root)
            self.tree.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

            # Define and format columns
            self.tree["columns"] = ("vecAng", "vecMagShear", "vecMagOrth", "vecRatio", "Level")
            self.tree.column("#0", width=0, stretch=tk.NO)
            self.tree.column("vecAng", anchor=tk.W, width=150)
            self.tree.column("vecMagShear", anchor=tk.W, width=150)
            self.tree.column("vecMagOrth", anchor=tk.W, width=150)
            self.tree.column("vecRatio", anchor=tk.W, width=150)
            self.tree.column("Level", anchor=tk.W, width=150)

            # Set column headings
            self.tree.heading("#0", text="", anchor=tk.W)
            self.tree.heading("vecAng", text="Vector Angle", anchor=tk.W)
            self.tree.heading("vecMagShear", text="Shear Vector Magnitude", anchor=tk.W)
            self.tree.heading("vecMagOrth", text="Normal Vector Magnitude", anchor=tk.W)
            self.tree.heading("vecRatio", text="Vector Ratio", anchor=tk.W)
            self.tree.heading("Level", text="Level", anchor=tk.W)

            # Insert data
            data = self.stored_data
            for i, row in enumerate(data):
                self.tree.insert(parent='', index='end', iid=i, text='', values=row)

    def restart_app(self):
        # Destroy the current window
        self.root.destroy()

        # Create a new root window and start a new instance of the application
        new_root = tk.Tk()
        new_app = ImageApp(new_root)
        new_root.mainloop()

    def delete_point(self):
        self.canvas.delete(self.point_dict[self.delete_p.get()])
        self.canvas.delete(self.text_dict[self.delete_p.get()])
        self.coordinates.pop(self.delete_p.get())

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
