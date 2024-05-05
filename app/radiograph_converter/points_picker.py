# Author: Alastair Quinn 2023

import vtk
import numpy as np

def getOriginalPointIndices(renderer):
    #print('start getOriginalPointIndices2')
    #print(f"n_actors: {len(renderer.actors.values())}")

    # temporarily increase resolution of HardwareSelector otherwise points are missed
    render_window = renderer.GetRenderWindow()
    size_init = render_window.GetSize()
    width, height = size_init[0], size_init[1]  # set the size of the render window
    gain = 6
    render_window.SetSize(width*gain, height*gain)

    selector = vtk.vtkHardwareSelector()
    selector.SetRenderer(renderer)
    selector.SetFieldAssociation(vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS)
    selector.SetArea(
        int(renderer.GetPickX1()*gain),
        int(renderer.GetPickY1()*gain),
        int(renderer.GetPickX2()*gain),
        int(renderer.GetPickY2()*gain))

    selected = selector.Select()
    #print(f"Number of nodes: {selected.GetNumberOfNodes()}")

    original_point_indices = []
    for i in range(selected.GetNumberOfNodes()):
        #print(f"nodes num: {i}")
        node = selected.GetNode(i)
        actor2 = node.GetProperties().Get(vtk.vtkSelectionNode.PROP())
        addr1  = str(actor2.memory_address).replace('Addr=','')
        
        for actor in renderer.actors.values():
            if actor.GetMapper() and actor.GetPickable():
                
                actor_addr = str(actor.memory_address).replace('Addr=','')
                if actor_addr == addr1:
                    #print("node start")
                    #print(node)
                    #print("node end")
                    extr = vtk.vtkExtractSelection()
                    extr.SetInputData(0, actor.GetMapper().GetInput())
                    temp = vtk.vtkSelection()
                    temp.AddNode(node)
                    extr.SetInputData(1, temp)
                    extr.Update()
                    output_dataset = extr.GetOutput()
                    original_point_indices.append([output_dataset.GetPointData().GetArray("vtkOriginalPointIds").GetValue(i) for i in range(output_dataset.GetNumberOfPoints())])
    #print(original_point_indices)
    
    render_window.SetSize(width, height)
    return original_point_indices

#TODO: change show code to work only with points and so multi mesh selection is supported.

def enable_points_picking(
    self,
    callback=None,
    show=True,
    style='wireframe',
    line_width=5,
    color='pink',
    **kwargs,
):
    import weakref, pyvista
    from pyvista import _vtk
    from pyvista.utilities import try_callback
    self_ = weakref.ref(self)

    # make sure to consistently use renderer
    renderer_ = weakref.ref(self.renderer)
    picker = _vtk.vtkRenderedAreaPicker()
    picker.GetSelectionPoint

    def end_pick_helper(picker, event_id):
        # Merge the selection into a single mesh
        picked = self_().picked_cells
        if isinstance(picked, pyvista.MultiBlock):
            if picked.n_blocks > 0:
                picked = picked.combine()
            else:
                picked = pyvista.UnstructuredGrid()
        # Check if valid
        is_valid_selection = picked.n_cells > 0

        if show and is_valid_selection:
            # Select the renderer where the mesh is added.
            active_renderer_index = self_().renderers._active_index
            for index in range(len(self.renderers)):
                renderer = self.renderers[index]
                for actor in renderer._actors.values():
                    mapper = actor.GetMapper()
                    if isinstance(mapper, _vtk.vtkDataSetMapper):
                        loc = self_().renderers.index_to_loc(index)
                        self_().subplot(*loc)
                        break

            # Use try in case selection is empty
            self_().add_mesh(
                picked,
                name='_point_picking_selection',
                style=style,
                color=color,
                line_width=line_width,
                pickable=False,
                reset_camera=False,
                **kwargs,
            )

            # Reset to the active renderer.
            loc = self_().renderers.index_to_loc(active_renderer_index)
            self_().subplot(*loc)

            # render here prior to running the callback
            self_().render()
        elif not is_valid_selection:
            self.remove_actor('_points_picking_selection')
            self_().picked_cells = None

        if callback is not None:
            try_callback(callback, self_().picked_points)

        # TODO: Deactivate selection tool
        return

    def through_pick_call_back(picker, event_id):
        self_().picked_points = []
        picked = pyvista.MultiBlock()
        for actor in renderer_().actors.values():
            if actor.GetMapper() and actor.GetPickable():
                input_mesh = pyvista.wrap(actor.GetMapper().GetInputAsDataSet())
                input_mesh.cell_data['orig_extract_id'] = np.arange(input_mesh.n_cells)
                extract = _vtk.vtkExtractGeometry()
                extract.SetInputData(input_mesh)
                extract.SetImplicitFunction(picker.GetFrustum())
                extract.Update()
                picked.append(pyvista.wrap(extract.GetOutput()))
        
        self_().picked_points = getOriginalPointIndices(renderer_())

        if len(picked) == 1:
            self_().picked_cells = picked[0]
        else:
            self_().picked_cells = picked
        return end_pick_helper(picker, event_id)

    self._picker = _vtk.vtkRenderedAreaPicker()
    self._picker.AddObserver(_vtk.vtkCommand.EndPickEvent, through_pick_call_back)

    self.enable_rubber_band_style()
    self.iren.set_picker(self._picker)
