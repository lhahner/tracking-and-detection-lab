import argparse                                            
              
from plugin_loader.plugin_installer import PluginInstaller
                                             

def handle_arguments(args):
    if args.plugin and args.install:
        PluginInstaller().install_plugin(name=args.name,
                                         path=args.path,
                                         git=args.git)
        breakpoint()

    else:
        raise ValueError("The provided argument combination is not supported.")
                                                           
def main():                                                
    parser = argparse.ArgumentParser(                      
        prog="tracking-and-detection-lab",                                 
        description="Tracking-and-detection-lab CLI entry point"                       
    )                                                      
                                                           
    parser.add_argument("plugin", help="Use this to install, update or change plugins for the application.")      
    parser.add_argument("install", help="Use this to install a new Plugin. Default mode install plugin from local path.")
    parser.add_argument("--name", help="the name of your plugin")
    parser.add_argument("--git", help="Installs plugin from git repository with public access.")
    parser.add_argument("--path", help="Local or Online path.")    
                                                           
    args = parser.parse_args()                             
    handle_arguments(args)
                                                           
if __name__ == "__main__":                                 
    main()                                                 