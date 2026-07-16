#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Track visualization module.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import math
import colorsys


class TrackVisualizer:
    """Track visualization."""

    def __init__(self,
                max_trajectory_points: int = 30,
                disappear_zone_width: int = 100,
                appear_zone_width: int = 100):
        """
        Initialize visualizer.

        Args:
            max_trajectory_points: Max trajectory points
            disappear_zone_width: Disappear zone width
            appear_zone_width: Appear zone width
        """
        self.max_trajectory_points = max_trajectory_points
        self.disappear_zone_width = disappear_zone_width
        self.appear_zone_width = appear_zone_width
        self.track_colors = {}
        self.static_tracks = set()
        self.trajectory_points = {}

    def generate_distinct_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate n distinct colors."""
        colors = []
        for i in range(n):
            h = i / n
            s = 0.7 + 0.3 * (i % 3) / 2
            v = 0.8 + 0.2 * ((i + 1) % 2)
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            colors.append((int(r * 255), int(g * 255), int(b * 255)))
        return colors

    def get_track_color(self, track_id: int, class_id: int = 0) -> Tuple[int, int, int]:
        """Get consistent color for track."""
        if track_id not in self.track_colors:
            base_colors = [
                (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0),
                (0, 255, 255), (255, 0, 255), (128, 0, 128), (255, 165, 0),
                (0, 128, 0), (128, 0, 0), (0, 0, 128)
            ]
            color_idx = (track_id + class_id * 7) % len(base_colors)
            self.track_colors[track_id] = base_colors[color_idx]
        return self.track_colors[track_id]

    def draw_tracks(self,
                  frame: np.ndarray,
                  tracks: List[Dict],
                  current_frame_id: int,
                  draw_matching_region: bool = False,
                  draw_trajectory: bool = True,
                  show_ids: bool = True,
                  show_class: bool = True,
                  line_thickness: int = 2) -> np.ndarray:
        """Draw tracks on frame."""
        self.draw_zones(frame)

        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            center_x, center_y = track['center_x'], track['center_y']
            is_static = track.get('is_static', False)
            is_lost = track.get('is_lost', False)
            class_id = track.get('class_id', 0)

            color = self.get_track_color(track_id, class_id)

            if is_static:
                self.static_tracks.add(track_id)
            else:
                self.static_tracks.discard(track_id)

            if track_id not in self.trajectory_points:
                self.trajectory_points[track_id] = []
            self.trajectory_points[track_id].append((int(center_x), int(center_y)))
            if len(self.trajectory_points[track_id]) > self.max_trajectory_points:
                self.trajectory_points[track_id] = self.trajectory_points[track_id][-self.max_trajectory_points:]

            box_color = color if not is_lost else (128, 128, 128)
            cv2.rectangle(frame,
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[2]), int(bbox[3])),
                        box_color,
                        line_thickness)

            label = ""
            if show_ids:
                label += f"ID:{track_id}"
            if show_class:
                label += f" C:{class_id}" if label else f"C:{class_id}"

            if label:
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame,
                            (int(bbox[0]), int(bbox[1] - label_size[1] - baseline)),
                            (int(bbox[0] + label_size[0]), int(bbox[1])),
                            box_color,
                            -1)
                cv2.putText(frame, label,
                           (int(bbox[0]), int(bbox[1] - baseline)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if draw_trajectory and len(self.trajectory_points[track_id]) > 1:
                path_points = np.array(self.trajectory_points[track_id], np.int32)
                path_points = path_points.reshape((-1, 1, 2))
                cv2.polylines(frame, [path_points], False, color, line_thickness)

            if draw_matching_region and not is_static:
                self.draw_matching_region(frame, track)

        return frame

    def draw_matching_region(self, frame: np.ndarray, track: Dict) -> None:
        """Draw matching region."""
        track_id = track['track_id']
        center_x, center_y = track['center_x'], track['center_y']
        direction = track.get('direction', 0.0)
        match_range = track.get('match_range', 150.0)
        color = self.get_track_color(track_id)

        if 135 <= direction <= 225:
            angle1_rad = math.radians(direction - 15)
            angle2_rad = math.radians(direction + 15)

            p1_x = center_x + match_range * math.cos(angle1_rad)
            p1_y = center_y + match_range * math.sin(angle1_rad)
            p2_x = center_x + match_range * math.cos(angle2_rad)
            p2_y = center_y + match_range * math.sin(angle2_rad)

            triangle_points = np.array([
                [int(center_x), int(center_y)],
                [int(p1_x), int(p1_y)],
                [int(p2_x), int(p2_y)]
            ], np.int32)

            overlay = frame.copy()
            cv2.fillPoly(overlay, [triangle_points], color)
            cv2.polylines(overlay, [triangle_points], True, color, 1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        else:
            region = (
                int(center_x - match_range),
                int(center_y - match_range / 2),
                int(center_x + match_range / 4),
                int(center_y + match_range / 2)
            )
            overlay = frame.copy()
            cv2.rectangle(overlay, (region[0], region[1]), (region[2], region[3]), color, -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            cv2.rectangle(frame, (region[0], region[1]), (region[2], region[3]), color, 1)

    def draw_zones(self, frame: np.ndarray) -> None:
        """Draw disappear and appear zones."""
        height, width = frame.shape[:2]

        cv2.rectangle(frame, (0, 0), (self.disappear_zone_width, height), (128, 0, 0), 2)
        cv2.putText(frame, "Disappear Zone", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 0), 2)

        cv2.rectangle(frame, (width - self.appear_zone_width, 0), (width, height), (0, 128, 0), 2)
        cv2.putText(frame, "Appear Zone", (width - self.appear_zone_width + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 0), 2)

    def visualize_track_stats(self,
                            tracks_history: List[Dict],
                            output_path: str,
                            figure_size: Tuple[int, int] = (12, 10)) -> None:
        """Visualize track statistics."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        tracks_data = {}
        for track in tracks_history:
            track_id = track['track_id']
            if track_id not in tracks_data:
                tracks_data[track_id] = []
            tracks_data[track_id].append(track)

        fig, axes = plt.subplots(2, 2, figsize=figure_size)
        fig.suptitle('Microbead Trajectory Analysis', fontsize=16)

        ax1 = axes[0, 0]
        for track_id, track_frames in tracks_data.items():
            frames = [t['frame_id'] for t in track_frames]
            x_positions = [t['center_x'] for t in track_frames]
            ax1.plot(frames, x_positions, '-', label=f'ID {track_id}')
        ax1.set_title('X Position vs. Time')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('X Position (pixels)')
        ax1.grid(True)

        ax2 = axes[0, 1]
        all_speeds = []
        for track_frames in tracks_data.values():
            for t in track_frames:
                if 'velocity_x' in t and 'velocity_y' in t:
                    speed = math.sqrt(t['velocity_x']**2 + t['velocity_y']**2)
                    all_speeds.append(speed)
        if all_speeds:
            ax2.hist(all_speeds, bins=20, alpha=0.7)
            ax2.set_title('Velocity Distribution')
            ax2.set_xlabel('Velocity (pixels/frame)')
            ax2.set_ylabel('Frequency')
            ax2.grid(True)

        ax3 = axes[1, 0]
        track_lengths = [len(track_frames) for track_frames in tracks_data.values()]
        if track_lengths:
            ax3.hist(track_lengths, bins=10, alpha=0.7)
            ax3.set_title('Track Length Distribution')
            ax3.set_xlabel('Track Length (frames)')
            ax3.set_ylabel('Frequency')
            ax3.grid(True)

        ax4 = axes[1, 1]
        track_start_frames = {tid: min(t['frame_id'] for t in frames) for tid, frames in tracks_data.items() if frames}
        if track_start_frames:
            min_start = min(track_start_frames.values())
            max_start = max(track_start_frames.values())
            span = max(1, max_start - min_start)
        else:
            min_start, span = 0, 1

        for track_id, track_frames in tracks_data.items():
            x_positions = [t['center_x'] for t in track_frames]
            y_positions = [t['center_y'] for t in track_frames]
            start_f = track_start_frames.get(track_id, min_start)
            norm = (start_f - min_start) / span
            alpha_val = 0.2 + 0.8 * norm
            ax4.plot(x_positions, y_positions, '-', label=f'ID {track_id}', alpha=alpha_val)
        ax4.set_title('Trajectory Path (X-Y)')
        ax4.set_xlabel('X Position (pixels)')
        ax4.set_ylabel('Y Position (pixels)')
        ax4.grid(True)

        if len(tracks_data) <= 10:
            ax1.legend(loc='upper right')
            ax4.legend(loc='upper right')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_path)
        plt.close()

    def create_tracking_video(self,
                            input_video_path: str,
                            output_video_path: str,
                            tracks_history: List[Dict],
                            draw_trajectory: bool = True,
                            draw_matching_region: bool = False,
                            line_thickness: int = 2) -> None:
        """Create tracking result video."""
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frames_tracks = {}
        for track in tracks_history:
            frame_id = track['frame_id']
            if frame_id not in frames_tracks:
                frames_tracks[frame_id] = []
            frames_tracks[frame_id].append(track)

        for frame_id in range(1, total_frames + 1):
            ret, frame = cap.read()
            if not ret:
                break

            current_tracks = frames_tracks.get(frame_id, [])

            if current_tracks:
                frame = self.draw_tracks(
                    frame, current_tracks, frame_id,
                    draw_matching_region=draw_matching_region,
                    draw_trajectory=draw_trajectory,
                    line_thickness=line_thickness
                )
            else:
                self.draw_zones(frame)

            cv2.putText(frame, f"Frame: {frame_id}/{total_frames}",
                      (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            out.write(frame)

        cap.release()
        out.release()

    def plot_trajectories(self,
                         tracks_history: List[Dict],
                         output_path: str,
                         figure_size: Tuple[int, int] = (12, 10)) -> None:
        """Plot all trajectories."""
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        tracks_data = {}
        for track in tracks_history:
            track_id = track['track_id']
            if track_id not in tracks_data:
                tracks_data[track_id] = []
            tracks_data[track_id].append(track)

        fig, ax = plt.subplots(figsize=figure_size)

        colors = self.generate_distinct_colors(len(tracks_data))
        color_map = {track_id: colors[i % len(colors)] for i, track_id in enumerate(tracks_data.keys())}

        for track_id, track_frames in tracks_data.items():
            track_frames.sort(key=lambda x: x['frame_id'])
            x_positions = [t['center_x'] for t in track_frames]
            y_positions = [t['center_y'] for t in track_frames]

            color_rgb = color_map[track_id]
            color = tuple(c / 255.0 for c in color_rgb)

            ax.plot(x_positions, y_positions, '-', color=color, linewidth=2,
                   label=f'ID {track_id}', alpha=0.7)

            if len(x_positions) > 0:
                ax.plot(x_positions[0], y_positions[0], 'o', color=color, markersize=6, alpha=0.8)
                ax.plot(x_positions[-1], y_positions[-1], 's', color=color, markersize=6, alpha=0.8)

        ax.set_xlabel('X Position (pixels)', fontsize=12)
        ax.set_ylabel('Y Position (pixels)', fontsize=12)
        ax.set_title('Microbead Trajectories', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        num_tracks = len(tracks_data)
        if num_tracks <= 30:
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, ncol=1, framealpha=0.9)
        else:
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 30:
                from matplotlib.lines import Line2D
                selected_handles = handles[:15] + handles[-15:]
                selected_labels = labels[:15] + labels[-15:]
                if len(handles) > 30:
                    selected_handles.insert(15, Line2D([0], [0], color='none'))
                    selected_labels.insert(15, f'... ({len(handles) - 30} more)')
                ax.legend(selected_handles, selected_labels,
                        bbox_to_anchor=(1.02, 1), loc='upper left',
                        fontsize=6, ncol=1, framealpha=0.9)
            else:
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, ncol=1, framealpha=0.9)

        try:
            try:
                plt.tight_layout(rect=[0, 0, 0.95, 1])
            except Exception:
                plt.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.1)

            plt.savefig(output_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none', format='png')

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"Trajectory plot saved to: {output_path}")
            else:
                raise Exception(f"Failed to save image: {output_path}")
        except Exception as e:
            print(f"Error saving trajectory plot: {e}")
            try:
                plt.subplots_adjust(left=0.15, right=0.7, top=0.95, bottom=0.1)
                plt.savefig(output_path, dpi=200, facecolor='white', format='png')
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    print(f"Trajectory plot saved (fallback): {output_path}")
                else:
                    raise Exception("Fallback save also failed")
            except Exception as e2:
                print(f"Error: fallback save also failed: {e2}")
                raise
        finally:
            plt.close(fig)
