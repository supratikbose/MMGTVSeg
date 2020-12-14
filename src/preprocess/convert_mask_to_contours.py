import vtk
from vtk.util import numpy_support
import glob
import numpy as np
import pydicom

class Slice:
    dcm_fname = ''
    uid = ''
    pos = []
    regions = None

def convert_mask_to_contours(fname, dicom_dir, label_pars, display = False):

    # get DICOM slices
    slice_pos, uid_list, fnames_list = get_dicom_slices(dicom_dir = dicom_dir)

    # -----------------------------------------------
    # read in a label volume from a NIFTI file
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(fname)
    reader.Update()

    print(slice_pos)
    q_form_matrix = reader.GetQFormMatrix()
    print(q_form_matrix)
    nii_z_offset = q_form_matrix.GetElement(2,3)
    print(nii_z_offset)

    #print(reader.GetNIFTIHeader())
    #exit()

    # define object connections within VTK processing pipeline
    threshold = vtk.vtkImageThreshold()
    threshold.SetInputConnection(reader.GetOutputPort())
    dmc = vtk.vtkDiscreteMarchingCubes()
    dmc.SetInputConnection(threshold.GetOutputPort())   
    smoother = vtk.vtkSmoothPolyDataFilter()    
    smoother.SetInputConnection(dmc.GetOutputPort())
    #decimate = vtk.vtkDecimatePro()
    #decimate.SetInputConnection(smoother.GetOutputPort())
    #decimate.SetTargetReduction(0.9)
    mapper = vtk.vtkPolyDataMapper()   
    mapper.SetInputConnection(smoother.GetOutputPort())
    plane_source = vtk.vtkPlaneSource()
    plane = vtk.vtkPlane()
    cutter = vtk.vtkCutter()    
    cutter.SetCutFunction(plane)
    
    #cutter.GenerateTrianglesOff()
    #cutter.SetSortByToSortByCell()	
    cutter.SetOutputPointsPrecision(1) # try double precision
    cutter.SetInputConnection(smoother.GetOutputPort())   
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputConnection(cutter.GetOutputPort())
    connectivity.SetOutputPointsPrecision(1)
    connectivity.ScalarConnectivityOff()
    stripper = vtk.vtkStripper()
    stripper.SetInputConnection(connectivity.GetOutputPort())

    cutPoly = vtk.vtkPolyData()
    cutBoundary = vtk.vtkFeatureEdges()    

    roi = []

    for label_num in range(1, label_pars['num_labels']+1):
        
        # threshold label mask volume to isolate ROI of interest
        threshold.ThresholdBetween(label_num, label_num)  
        threshold.SetInValue(1)  
        threshold.SetOutValue(0)
        threshold.ReplaceInOn()
        threshold.ReplaceOutOn()    
        
        # convert label mask to surface mesh
        dmc.GenerateValues(1, 1, 1)
        dmc.Update()

        # smooth surface mesh to improve visual appearance
        smoother.SetNumberOfIterations(label_pars['label_smoothing'][str(label_num)])
        #smoother.SetNumberOfIterations(20)
        #smoother.SetFeatureAngle(110)
        #smoother.SetRelaxationFactor(0.01)
        #smoother.FeatureEdgeSmoothingOn()
        #smoother.BoundarySmoothingOn()
        smoother.SetOutputPointsPrecision(1)  # try double precision float

        # extract structure regions based resulting regions
        connectivity.SetExtractionModeToSpecifiedRegions()
        
        # determine structure bounding box to only slice through relevant parts of volume
        structure_bounding_box = mapper.GetBounds()
        # structure_bounding_box[4] += nii_z_offset
        # structure_bounding_box[5] += nii_z_offset        
        print('Structure: ', label_pars['label_names'][str(label_num)])
        print('Bounding box: ', structure_bounding_box)
        z_indeces = np.where((slice_pos[:,2] >= structure_bounding_box[4]+nii_z_offset) & (slice_pos[:,2] <= structure_bounding_box[5]+nii_z_offset))
        #z_indeces = list(np.array([75]))

        # HN007 brainstem Z bounding box should be zmin: 312.5 , zmax: 375.5

        ss = []
        
        for z in z_indeces[0]:
            #print(z_indeces[0])
            #exit()
        #for z in z_indeces:
            
            # cut structure with current plane
            print('slice_pos[z]: ', slice_pos[z])      
            plane_source.SetCenter(0, 0, slice_pos[z][2]-nii_z_offset)
            plane_source.SetPoint1(500, 0, slice_pos[z][2]-nii_z_offset)
            plane_source.SetPoint2(0, 500, slice_pos[z][2]-nii_z_offset)
            plane_source.SetOrigin(0, 0, slice_pos[z][2]-nii_z_offset)
            
            plane_source.SetNormal(0, 0, 1)
            plane_source.Update()
            plane.SetOrigin(plane_source.GetOrigin())
            plane.SetNormal(plane_source.GetNormal())
            smoother.Update()
            #decimate.Update()
            cutter.Update()
            
            # extract contour regions
            connectivity.InitializeSpecifiedRegionList()
            connectivity.AddSpecifiedRegion(0)
            connectivity.Modified()
            connectivity.Update()
            num_regions = connectivity.GetNumberOfExtractedRegions()
            #print('Number of extracted regions: ', num_regions)

            if (num_regions > 0):
                regions = []

                # -----------------
                stripper.SetMaximumLength(10000)
                stripper.JoinContiguousSegmentsOn()
                stripper.Update()
                # p = stripper.GetOutput().GetPoints()
                # test
                cutPoly.SetPoints(stripper.GetOutput().GetPoints())
                cutPoly.SetPolys(stripper.GetOutput().GetLines())
                cutBoundary.SetInputData(cutPoly)
                cutBoundary.Update()
                p = cutBoundary.GetOutput().GetPoints()               
                
                if (p is not None):

                    p_d = p.GetData()
                    p_np = numpy_support.vtk_to_numpy(p_d)
                    #lines = stripper.GetOutput().GetLines()
                    #test
                    lines = cutBoundary.GetOutput().GetLines()

                    d = lines.GetData()
                    d_np = numpy_support.vtk_to_numpy(d)
                    d_np = np.delete(d_np, np.arange(0, d_np.size, 3))
                    pp = p_np[d_np, 0:3]
                    pp[:, 0] += slice_pos[z][0] + 0.5
                    pp[:, 1] += slice_pos[z][1] + 0.5
                    pp[:, 2] += nii_z_offset
                    regions.append( pp )
                    current_slice = Slice()
                    current_slice.dcm_fname = fnames_list[z]
                    current_slice.pos = slice_pos[z]
                    current_slice.regions = regions
                    current_slice.uid = uid_list[z]
                    ss.append(current_slice)

                    for i in range(1, num_regions):
                        connectivity.SetExtractionModeToSpecifiedRegions()
                        connectivity.InitializeSpecifiedRegionList()
                        connectivity.AddSpecifiedRegion(i)
                        connectivity.Modified()
                        connectivity.Update()                
                        stripper.Update()
                        
                        # p = stripper.GetOutput().GetPoints()
                        # test
                        cutPoly.SetPoints(stripper.GetOutput().GetPoints())
                        cutPoly.SetPolys(stripper.GetOutput().GetLines())
                        cutBoundary.SetInputData(cutPoly)
                        cutBoundary.Update()
                        p = cutBoundary.GetOutput().GetPoints()

                        if (p is not None):
                            regions = []
                            p_d = p.GetData()
                            p_np = numpy_support.vtk_to_numpy(p_d)                            
                            #lines = stripper.GetOutput().GetLines()
                            #test
                            lines = cutBoundary.GetOutput().GetLines()                           

                            d = lines.GetData()
                            d_np = numpy_support.vtk_to_numpy(d)
                            d_np = np.delete(d_np, np.arange(0, d_np.size, 3))
                            pp = p_np[d_np, 0:3]
                            pp[:, 0] += slice_pos[z][0] + 0.5
                            pp[:, 1] += slice_pos[z][1] + 0.5
                            pp[:, 2] += nii_z_offset
                            regions.append( pp )
                            current_slice = Slice()
                            current_slice.dcm_fname = fnames_list[z]
                            current_slice.pos = slice_pos[z]
                            current_slice.regions = regions
                            current_slice.uid = uid_list[z]
                            ss.append(current_slice)

        roi.append(ss)


        if display == True:

            plane_mapper = vtk.vtkPolyDataMapper()
            plane_mapper.SetInputConnection(plane_source.GetOutputPort())
            plane_actor = vtk.vtkActor()
            plane_actor.SetMapper(plane_mapper)
            plane_actor.GetProperty().SetColor(0.5, 0.5, 0.5)
            plane_actor.GetProperty().SetOpacity(0.7)
            
            # mapper = vtk.vtkPolyDataMapper()   
            # mapper.SetInputConnection(smoother.GetOutputPort())
            mapper.ScalarVisibilityOff()
            vtk.vtkPolyDataMapper().SetResolveCoincidentTopologyToPolygonOffset()

            actor = vtk.vtkActor()
            actor.GetProperty().SetColor(0.5, 0.0, 0.0)
            actor.GetProperty().SetOpacity(0.7)
            #actor.GetProperty().EdgeVisibilityOn()
            actor.SetMapper(mapper)

            cutter_mapper = vtk.vtkPolyDataMapper()
            cutter_mapper.SetInputConnection(connectivity.GetOutputPort())
            cutter_actor = vtk.vtkActor()
            cutter_actor.GetProperty().SetColor(1.0,1,0)
            cutter_actor.GetProperty().SetLineWidth(2)
            cutter_actor.SetMapper(cutter_mapper)

            # set up rendering
            renderer = vtk.vtkRenderer()
            renderer.AddActor(actor)
            renderer.AddActor(plane_actor)
            renderer.AddActor(cutter_actor)
            #renderer.SetBackground(background_color)
            camera = renderer.MakeCamera()
            renderer.SetActiveCamera(camera)
            renWin = vtk.vtkRenderWindow()
            renWin.AddRenderer(renderer)
            renWin.SetSize(800, 600)
            iren = vtk.vtkRenderWindowInteractor()
            iren.SetRenderWindow(renWin)

            # turn on depth peeling technique to improve visualization of translucent objects
            renderer.SetUseDepthPeeling(1)
            renWin.SetAlphaBitPlanes(1)
            renWin.SetMultiSamples(0)
            # Set interactor style to "trackball"
            inStyle = vtk.vtkInteractorStyleSwitch()
            iren.SetInteractorStyle(inStyle)
            iren.SetKeyEventInformation(0, 0, 't', 0, '0')
            iren.InvokeEvent("CharEvent")
            iren.Initialize()
            renderer.ResetCamera()
            renWin.Render()
            # Start the event loop.
            iren.Start()

    return roi, uid_list, fnames_list

def get_dicom_slices(dicom_dir):
    fname_list = sorted(glob.glob(dicom_dir + '//*.dcm'))    
    pos = []
    z = []
    uid = []
    for fname in fname_list:
        #print(fname)
        dcm = pydicom.dcmread(fname)
        if dcm.SOPClassUID == '1.2.840.10008.5.1.4.1.1.2':      # make sure file is a CT slice
            print(dcm.ImagePositionPatient)
            pos.append(dcm.ImagePositionPatient)
            uid.append(dcm.SOPInstanceUID)
        else:
            fname_list.remove(fname)        
    pos = np.asarray(pos)
    sorted_indeces = np.argsort(pos[:,2], axis=0)
    pos = pos[sorted_indeces, :]
    fname_list = [fname_list[i] for i in sorted_indeces]
    #print(pos)
    #print(fname_list)
    return pos, uid, fname_list
